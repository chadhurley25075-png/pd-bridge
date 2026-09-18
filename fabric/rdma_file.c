#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
/* rdma_file.c — move one file Spark -> Mac (or Mac -> Spark) with RDMA WRITE over MCDMA/RoCE.
 * Same wire protocol both roles; stdout handshake like Ash's tools.
 *   recv: rdma_file recv DEV GID_IDX OUT_PATH NBYTES        -> prints "EP gid qpn psn rkey addr", waits, lands, prints RESULT
 *   send: rdma_file send DEV GID_IDX IN_PATH  < EP-line     -> prints "EP gid qpn psn", RDMA-WRITEs the file, prints RESULT
 * Region is registered as a pool of <=4 MiB MRs on the Mac (MCDMA translator cap); sender streams 1 MiB WRITEs,
 * depth 7, then a final WRITE_WITH_IMM as the completion marker. Compass 2026-09-18, Apache-2.0. */
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#define CHUNK (1u<<20)
#define REGION (4u<<20)      /* per-MR size (Mac cap) */
#define WINDOW (56u<<20)     /* 2 slots = 112 MiB registered */
#define SLOTS 2
#define DEPTH 14
#define PSN 0x11223u
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void die(const char*w){fprintf(stderr,"rdma_file: %s: %s\n",w,strerror(errno));exit(2);}
static struct ibv_context*open_dev(const char*n){int c=0;struct ibv_device**l=ibv_get_device_list(&c);struct ibv_context*x=NULL;for(int i=0;i<c;i++)if(!strcmp(ibv_get_device_name(l[i]),n))x=ibv_open_device(l[i]);ibv_free_device_list(l);if(!x)die("open device");return x;}
int main(int argc,char**argv){
    if(argc<5){fprintf(stderr,"usage: rdma_file recv|send DEV GID_IDX PATH [NBYTES]\n");return 2;}
    const int recv=!strcmp(argv[1],"recv"); const char*dev=argv[2]; unsigned gidx=(unsigned)atoi(argv[3]); const char*path=argv[4];
    setvbuf(stdout,NULL,_IOLBF,0);
    struct ibv_context*ctx=open_dev(dev); struct ibv_port_attr port; if(ibv_query_port(ctx,1,&port))die("query port");
    if(port.state!=IBV_PORT_ACTIVE){fprintf(stderr,"port not active\n");return 2;}
    enum ibv_mtu mtu=port.active_mtu>IBV_MTU_4096?IBV_MTU_4096:port.active_mtu;
    union ibv_gid gid; if(ibv_query_gid(ctx,1,gidx,&gid))die("query gid"); char mygid[64]; inet_ntop(AF_INET6,&gid,mygid,64);
    struct ibv_pd*pd=ibv_alloc_pd(ctx); if(!pd)die("pd");
    /* MCDMA grants small CQs/QPs; back off like Ash's bench does */
    struct ibv_cq*cq=NULL; for(int want=31;want>=4&&!cq;want/=2) cq=ibv_create_cq(ctx,want,NULL,NULL,0); if(!cq)die("cq");
    struct ibv_qp_init_attr ia; memset(&ia,0,sizeof ia); ia.send_cq=cq; ia.recv_cq=cq; ia.qp_type=IBV_QPT_RC;
    ia.cap.max_send_sge=1; ia.cap.max_recv_sge=1;
    struct ibv_qp*qp=NULL; for(unsigned wr=30;wr>=8&&!qp;wr/=2){ ia.cap.max_send_wr=wr; ia.cap.max_recv_wr=wr; qp=ibv_create_qp(pd,&ia);} if(!qp)die("create qp");
    struct ibv_qp_attr a; memset(&a,0,sizeof a); a.qp_state=IBV_QPS_INIT; a.port_num=1; a.qp_access_flags=IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_WRITE;
    if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))die("INIT");

    uint64_t nbytes; unsigned char*buf; size_t rsz; struct ibv_mr**mrs; unsigned nmr;
    if(recv){ nbytes=strtoull(argv[5],NULL,10); }
    else { struct stat st; if(stat(path,&st))die("stat"); nbytes=(uint64_t)st.st_size; }
    uint64_t slot_bytes; { uint64_t need=nbytes+64; if(need>WINDOW) need=WINDOW; slot_bytes=(need+REGION-1)/REGION*REGION; }
    rsz=(size_t)(slot_bytes*SLOTS); nmr=(unsigned)(rsz/REGION); const unsigned mr_per_slot=(unsigned)(slot_bytes/REGION); const uint64_t cap=slot_bytes-64;
    if(posix_memalign((void**)&buf,16384,rsz))die("alloc"); memset(buf,0,rsz);
    int infd=-1; if(!recv){ infd=open(path,O_RDONLY); if(infd<0)die("open in"); }
    mrs=calloc(nmr,sizeof*mrs);
    for(unsigned i=0;i<nmr;i++){ mrs[i]=ibv_reg_mr(pd,buf+(size_t)i*REGION,REGION,IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE); if(!mrs[i])die("reg_mr"); }

    if(recv){
        /* completion = flag word at end of the last region (Ash's proven MCDMA pattern; no recv needed) */
        /* advertise every region: EP gid qpn psn nmr then one line per MR "rkey addr" */
        printf("EP %s %u %u %u\n",mygid,qp->qp_num,PSN,nmr); for(unsigned i=0;i<nmr;i++) printf("MR %u %llu\n",mrs[i]->rkey,(unsigned long long)(uintptr_t)mrs[i]->addr);
        char rg[64]; unsigned rqpn,rpsn; if(scanf("EP %63s %u %u",rg,&rqpn,&rpsn)!=3)die("read peer EP");
        memset(&a,0,sizeof a); a.qp_state=IBV_QPS_RTR; a.path_mtu=mtu; a.dest_qp_num=rqpn; a.rq_psn=rpsn; a.max_dest_rd_atomic=1; a.min_rnr_timer=12;
        a.ah_attr.is_global=1; a.ah_attr.port_num=1; a.ah_attr.grh.sgid_index=gidx; a.ah_attr.grh.hop_limit=64; inet_pton(AF_INET6,rg,&a.ah_attr.grh.dgid);
        if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER))die("RTR");
        memset(&a,0,sizeof a); a.qp_state=IBV_QPS_RTS; a.sq_psn=PSN; a.timeout=14; a.retry_cnt=7; a.rnr_retry=7; a.max_rd_atomic=1;
        if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC))die("RTS");
        volatile uint64_t*flags[SLOTS]; for(unsigned k=0;k<SLOTS;k++){ flags[k]=(volatile uint64_t*)(buf+(k+1)*slot_bytes-64); *flags[k]=0; } __atomic_thread_fence(__ATOMIC_SEQ_CST);
        int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0644); if(fd<0)die("open out"); const int sink=!strcmp(path,"/dev/null");
        printf("READY %llu %u\n",(unsigned long long)cap,SLOTS); double t0=now(); double twire=0;
        uint64_t done=0; unsigned seq=1;
        while(done<nbytes){ uint64_t wl=nbytes-done; if(wl>cap) wl=cap; unsigned k=(seq-1)%SLOTS; volatile uint64_t*flag=flags[k]; unsigned char*wb=buf+k*slot_bytes;
            double tw=now(); while(*flag!=seq){ if(now()-tw>600){fprintf(stderr,"timeout window %u\n",seq);return 2;} usleep(20); } __atomic_thread_fence(__ATOMIC_SEQ_CST); twire+=now()-tw;
            if(!sink){ uint64_t o=0; while(o<wl){ssize_t r=write(fd,wb+o,wl-o); if(r<=0)die("write"); o+=(uint64_t)r;} }
            *flag=0; __atomic_thread_fence(__ATOMIC_SEQ_CST); printf("ACK %u\n",seq); fflush(stdout);
            done+=wl; seq++; }
        close(fd); double t=now()-t0;
        fprintf(stderr,"wire_only_gbit=%.3f\n",nbytes*8/twire/1e9);
        printf("RESULT role=recv bytes=%llu seconds=%.6f gbit=%.3f\n",(unsigned long long)nbytes,t,nbytes*8/t/1e9);
    } else {
        char rg[64]; unsigned rqpn,rpsn,rnmr; if(scanf("EP %63s %u %u %u",rg,&rqpn,&rpsn,&rnmr)!=4)die("read peer EP");
        if(rnmr!=nmr){fprintf(stderr,"region count mismatch %u vs %u\n",rnmr,nmr);return 2;}
        uint32_t*rkey=calloc(nmr,4); uint64_t*raddr=calloc(nmr,8);
        for(unsigned i=0;i<nmr;i++){ unsigned long long ad; char ln[128]; if(!fgets(ln,sizeof ln,stdin)&&!fgets(ln,sizeof ln,stdin))die("read MR"); if(sscanf(ln,"MR %u %llu",&rkey[i],&ad)!=2){ if(!fgets(ln,sizeof ln,stdin)||sscanf(ln,"MR %u %llu",&rkey[i],&ad)!=2) die("parse MR"); } raddr[i]=ad; }
        printf("EP %s %u %u\n",mygid,qp->qp_num,PSN);
        memset(&a,0,sizeof a); a.qp_state=IBV_QPS_RTR; a.path_mtu=mtu; a.dest_qp_num=rqpn; a.rq_psn=rpsn; a.max_dest_rd_atomic=1; a.min_rnr_timer=12;
        a.ah_attr.is_global=1; a.ah_attr.port_num=1; a.ah_attr.grh.sgid_index=gidx; a.ah_attr.grh.hop_limit=64; inet_pton(AF_INET6,rg,&a.ah_attr.grh.dgid);
        if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER))die("RTR");
        memset(&a,0,sizeof a); a.qp_state=IBV_QPS_RTS; a.sq_psn=PSN; a.timeout=14; a.retry_cnt=7; a.rnr_retry=7; a.max_rd_atomic=1;
        if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC))die("RTS");
        char line[16]; unsigned long long capll; unsigned rslots; if(scanf("%15s %llu %u",line,&capll,&rslots)!=3||strcmp(line,"READY")||rslots!=SLOTS)die("peer not READY"); const uint64_t cap=capll; unsigned acked=0;
        double t0=now(); struct ibv_wc wc; uint64_t sent=0; unsigned seq=1; double tread=0,tburst=0,tack=0;
        while(sent<nbytes){ uint64_t wl=nbytes-sent; if(wl>cap) wl=cap; unsigned k=(seq-1)%SLOTS; unsigned base_mr=k*mr_per_slot;
        if(seq>SLOTS){ while(acked<seq-SLOTS){ double ta=now(); char ak[32]; unsigned as; if(scanf("%31s %u",ak,&as)!=2||strcmp(ak,"ACK"))die("bad ACK"); acked=as; tack+=now()-ta; } }
        unsigned char*sb=buf+k*slot_bytes; if(!getenv("RF_NOREAD")){ double tr=now(); uint64_t o=0; while(o<wl){ ssize_t r=read(infd,sb+o,wl-o); if(r<=0)die("read"); o+=(uint64_t)r; } tread+=now()-tr; }
        double tb=now(); uint64_t off=0; unsigned inflight=0; uint64_t lim=(wl+CHUNK-1)/CHUNK*CHUNK; if(lim>cap) lim=cap;
        while(off<lim||inflight){
            while(off<lim&&inflight<DEPTH){ unsigned i=base_mr+(unsigned)(off/REGION); uint64_t inreg=off%REGION; unsigned len=CHUNK; if(inreg+len>REGION)len=(unsigned)(REGION-inreg); if(off+len>lim)len=(unsigned)(lim-off);
                struct ibv_sge s={.addr=(uintptr_t)(sb+off),.length=len,.lkey=mrs[i]->lkey};
                struct ibv_send_wr w; memset(&w,0,sizeof w); w.wr_id=off; w.sg_list=&s; w.num_sge=1; w.opcode=IBV_WR_RDMA_WRITE; w.send_flags=IBV_SEND_SIGNALED;
                w.wr.rdma.remote_addr=raddr[i]+inreg; w.wr.rdma.rkey=rkey[i]; struct ibv_send_wr*bad; if(ibv_post_send(qp,&w,&bad))die("post_send"); off+=len; inflight++; }
            int n=ibv_poll_cq(cq,1,&wc); if(n<0)die("poll"); if(n){ if(wc.status!=IBV_WC_SUCCESS){fprintf(stderr,"wc status=%d\n",wc.status);return 2;} inflight--; }
        }
        /* window marker: write seq into the flag word */
        { uint64_t*fl=(uint64_t*)(sb+slot_bytes-64); *fl=seq; unsigned li=base_mr+mr_per_slot-1;
          struct ibv_sge s={.addr=(uintptr_t)fl,.length=8,.lkey=mrs[li]->lkey}; struct ibv_send_wr w; memset(&w,0,sizeof w); w.wr_id=~0ull; w.sg_list=&s; w.num_sge=1;
          w.opcode=IBV_WR_RDMA_WRITE; w.send_flags=IBV_SEND_SIGNALED; w.wr.rdma.remote_addr=raddr[li]+(REGION-64); w.wr.rdma.rkey=rkey[li];
          struct ibv_send_wr*bad; if(ibv_post_send(qp,&w,&bad))die("post flag"); int n; do{n=ibv_poll_cq(cq,1,&wc);}while(n==0); if(wc.status!=IBV_WC_SUCCESS){fprintf(stderr,"flag wc=%d\n",wc.status);return 2;} }
        tburst+=now()-tb;
        sent+=wl; seq++; }
        double t=now()-t0; fprintf(stderr,"send_profile read=%.3f burst=%.3f ack=%.3f wire_burst_gbit=%.2f\n",tread,tburst,tack,nbytes*8/tburst/1e9); printf("RESULT role=send bytes=%llu seconds=%.6f gbit=%.3f\n",(unsigned long long)nbytes,t,nbytes*8/t/1e9);
    }
    return 0;
}
