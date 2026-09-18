/* tbsend.c — file transfer over Apple RDMA-over-Thunderbolt (UC QP, IBV_WR_SEND only, per TN3205).
 * receiver: tbsend recv DEVICE GID_IDX OUTFILE            -> prints "ENDPOINT gid qpn psn" then waits
 * sender:   tbsend send DEVICE GID_IDX INFILE < endpoint  -> connects, streams file in <=16MB chunks
 * Both print TBRESULT bytes=.. seconds=.. gbit=.. ; receiver also prints sha256 prefix.
 * Compass 2026-09-18. Apache-2.0. */
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
#include <CommonCrypto/CommonDigest.h>

#define CHUNK (16u<<20)      /* <= 16,773,120 per TN3205; 16 MiB chunk uses 16,777,216 -> use 16,773,120 */
static unsigned MSG_MAX=256u<<10; /* frames; env TB_FRAME (bytes). TN3205: sender msg == receiver buffer size */
static unsigned RING=12;          /* env TB_RING. posted total must stay under the ~4 MiB TB RX ring */
#define PSN 0x123456u

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void die(const char*w){fprintf(stderr,"tbsend: %s: %s\n",w,strerror(errno));exit(2);}

static struct ibv_context* open_dev(const char*name){
    int n=0;struct ibv_device**l=ibv_get_device_list(&n);struct ibv_context*c=NULL;
    for(int i=0;i<n;i++) if(!strcmp(ibv_get_device_name(l[i]),name)) c=ibv_open_device(l[i]);
    ibv_free_device_list(l); if(!c) die("open device"); return c;
}
static struct ibv_qp* make_uc_qp(struct ibv_pd*pd,struct ibv_cq*cq){
    struct ibv_qp_init_attr ia; memset(&ia,0,sizeof ia);
    ia.send_cq=cq; ia.recv_cq=cq; ia.qp_type=IBV_QPT_UC;
    ia.cap.max_send_wr=RING*2+8; ia.cap.max_recv_wr=RING*2+8; ia.cap.max_send_sge=1; ia.cap.max_recv_sge=1;
    struct ibv_qp*qp=ibv_create_qp(pd,&ia); if(!qp) die("create qp");
    struct ibv_qp_attr a; memset(&a,0,sizeof a); a.qp_state=IBV_QPS_INIT; a.port_num=1; a.qp_access_flags=0;
    if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) die("qp INIT");
    return qp;
}
static void connect_uc(struct ibv_qp*qp,unsigned gid_idx,const char*rgid,uint32_t rqpn,uint32_t rpsn,uint16_t rlid,enum ibv_mtu mtu){
    struct ibv_qp_attr a; memset(&a,0,sizeof a);
    a.qp_state=IBV_QPS_RTR; a.path_mtu=mtu; a.dest_qp_num=rqpn; a.rq_psn=rpsn;
    a.ah_attr.dlid=rlid; a.ah_attr.sl=0; a.ah_attr.src_path_bits=0; a.ah_attr.is_global=1; a.ah_attr.port_num=1; a.ah_attr.grh.sgid_index=gid_idx; a.ah_attr.grh.hop_limit=1;
    if(inet_pton(AF_INET6,rgid,&a.ah_attr.grh.dgid)!=1) die("parse remote gid");
    if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN)) die("qp RTR");
    memset(&a,0,sizeof a); a.qp_state=IBV_QPS_RTS; a.sq_psn=PSN;
    if(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_SQ_PSN)) die("qp RTS");
}
static void gid_str(struct ibv_context*c,unsigned idx,char*out){
    union ibv_gid g; if(ibv_query_gid(c,1,idx,&g)) die("query gid"); inet_ntop(AF_INET6,&g,out,INET6_ADDRSTRLEN);
}

int main(int argc,char**argv){
    if(argc<5){fprintf(stderr,"usage: tbsend recv|send DEVICE GID_IDX FILE\n");return 2;}
    const int recv=!strcmp(argv[1],"recv"); const char*dev=argv[2]; unsigned gid_idx=(unsigned)atoi(argv[3]); const char*path=argv[4];
    setvbuf(stdout,NULL,_IOLBF,0);
    if(getenv("TB_FRAME")) MSG_MAX=(unsigned)atoi(getenv("TB_FRAME")); if(getenv("TB_RING")) RING=(unsigned)atoi(getenv("TB_RING"));
    struct ibv_context*ctx=open_dev(dev);
    struct ibv_port_attr port; if(ibv_query_port(ctx,1,&port)) die("query port");
    if(port.state!=IBV_PORT_ACTIVE){fprintf(stderr,"port not active\n");return 2;}
    enum ibv_mtu mtu=port.active_mtu;
    struct ibv_pd*pd=ibv_alloc_pd(ctx); if(!pd) die("pd");
    struct ibv_cq*cq=ibv_create_cq(ctx,(int)RING*4+16,NULL,NULL,0); if(!cq) die("cq");
    struct ibv_qp*qp=make_uc_qp(pd,cq);
    char mygid[INET6_ADDRSTRLEN]; gid_str(ctx,gid_idx,mygid);

    if(recv){
        /* header first: sender tells us total size in the first 8-byte message */
        unsigned char**ring=calloc(RING,sizeof*ring); struct ibv_mr**mr=calloc(RING,sizeof*mr);
        for(unsigned i=0;i<RING;i++){ if(posix_memalign((void**)&ring[i],16384,MSG_MAX)) die("alloc"); memset(ring[i],0,MSG_MAX);
            mr[i]=ibv_reg_mr(pd,ring[i],MSG_MAX,IBV_ACCESS_LOCAL_WRITE); if(!mr[i]) die("reg_mr recv"); }
        printf("ENDPOINT %s %u %u %u\n",mygid,qp->qp_num,PSN,port.lid);
        char rgid[64]; unsigned rqpn,rpsn,rlid; if(scanf("%63s %u %u %u",rgid,&rqpn,&rpsn,&rlid)!=4) die("read sender endpoint");
        connect_uc(qp,gid_idx,rgid,rqpn,rpsn,(uint16_t)rlid,mtu);
        for(unsigned i=0;i<RING;i++){ struct ibv_sge s={.addr=(uintptr_t)ring[i],.length=MSG_MAX,.lkey=mr[i]->lkey};
            struct ibv_recv_wr w={.wr_id=(uint64_t)i,.sg_list=&s,.num_sge=1},*bad; if(ibv_post_recv(qp,&w,&bad)) die("post_recv"); }
        printf("READY\n");
        const int sink=!strcmp(path,"/dev/null"); FILE*f=sink?NULL:fopen(path,"wb"); if(!sink&&!f) die("open out");
        CC_SHA256_CTX sha; CC_SHA256_Init(&sha);
        uint64_t total=0,expect=UINT64_MAX; double t0=0; int got_hdr=0;
        while(total<expect){
            struct ibv_wc wc; int n; do{ n=ibv_poll_cq(cq,1,&wc);}while(n==0);
            if(n<0||wc.status!=IBV_WC_SUCCESS){fprintf(stderr,"recv wc status=%d\n",wc.status);return 2;}
            int i=(int)wc.wr_id; unsigned len=wc.byte_len;
            if(!got_hdr){ memcpy(&expect,ring[i],8); got_hdr=1; t0=now(); }
            else { unsigned keep=(expect-total)<len?(unsigned)(expect-total):len; if(!sink){fwrite(ring[i],1,keep,f); CC_SHA256_Update(&sha,ring[i],keep);} total+=keep; }
            struct ibv_sge s={.addr=(uintptr_t)ring[i],.length=MSG_MAX,.lkey=mr[i]->lkey};
            struct ibv_recv_wr w={.wr_id=(uint64_t)i,.sg_list=&s,.num_sge=1},*bad; if(ibv_post_recv(qp,&w,&bad)) die("repost");
        }
        double t=now()-t0; if(f) fclose(f);
        unsigned char d[32]; CC_SHA256_Final(d,&sha);
        printf("TBRESULT role=recv bytes=%llu seconds=%.6f gbit=%.3f sha=",(unsigned long long)total,t,total*8/t/1e9);
        for(int i=0;i<8;i++) printf("%02x",d[i]); printf("\n");
        return 0;
    } else {
        struct stat st; if(stat(path,&st)) die("stat"); uint64_t size=(uint64_t)st.st_size;
        size_t rsz=(size_t)((size+MSG_MAX-1)&~(uint64_t)(MSG_MAX-1)); unsigned char*buf; if(posix_memalign((void**)&buf,16384,rsz)) die("alloc"); memset(buf,0,rsz); 
        FILE*f=fopen(path,"rb"); if(!f||fread(buf,1,size,f)!=size) die("read file"); fclose(f);
        struct ibv_mr*mr=ibv_reg_mr(pd,buf,rsz,0); if(!mr) die("reg_mr send");
        unsigned char*hbuf; if(posix_memalign((void**)&hbuf,16384,MSG_MAX)) die("alloc hdr"); memset(hbuf,0,MSG_MAX); memcpy(hbuf,&size,8);
        struct ibv_mr*hmr=ibv_reg_mr(pd,hbuf,MSG_MAX,0); if(!hmr) die("reg_mr hdr");
        char rgid[64]; unsigned rqpn,rpsn,rlid; if(scanf("ENDPOINT %63s %u %u %u",rgid,&rqpn,&rpsn,&rlid)!=4) die("read receiver endpoint");
        printf("%s %u %u %u\n",mygid,qp->qp_num,PSN,port.lid);
        connect_uc(qp,gid_idx,rgid,rqpn,rpsn,(uint16_t)rlid,mtu);
        char line[16]; if(scanf("%15s",line)!=1||strcmp(line,"READY")) die("receiver not ready");
        double t0=now(); unsigned inflight=0; uint64_t off=0; struct ibv_wc wc;
        /* header */
        { struct ibv_sge s={.addr=(uintptr_t)hbuf,.length=MSG_MAX,.lkey=hmr->lkey};
          struct ibv_send_wr w={.wr_id=0,.sg_list=&s,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
          if(ibv_post_send(qp,&w,&bad)) die("post hdr"); inflight++; }
        while(off<rsz||inflight){
            while(off<rsz&&inflight<RING){ unsigned len=MSG_MAX;
                struct ibv_sge s={.addr=(uintptr_t)(buf+off),.length=len,.lkey=mr->lkey};
                struct ibv_send_wr w={.wr_id=off,.sg_list=&s,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
                if(ibv_post_send(qp,&w,&bad)) die("post_send"); off+=len; inflight++; }
            int n=ibv_poll_cq(cq,1,&wc); if(n<0) die("poll");
            if(n){ if(wc.status!=IBV_WC_SUCCESS){fprintf(stderr,"send wc status=%d\n",wc.status);return 2;} inflight--; }
        }
        double t=now()-t0;
        printf("TBRESULT role=send bytes=%llu seconds=%.6f gbit=%.3f\n",(unsigned long long)size,t,size*8/t/1e9);
        return 0;
    }
}
