# pd_rdma — capture transport for pd-bridge (R1 pull, R2 push; docs/RDMA.md). Same sources on both boxes:
#   Spark:  make    -> pd_rdma (serve) + libpd_rdma_tx.so (loaded by spark/pd_kv_connector.py)   [rdma-core]
#   Mac:    make    -> pd_rdma (client / recvd), signed with the DriverKit userclient entitlement [MelonDMA]
# On the Mac a rebuild strips the entitlement unless it is re-signed; this Makefile always signs.

UNAME := $(shell uname)
CFLAGS ?= -std=c11 -O2 -Wall -Wextra

ifeq ($(UNAME),Darwin)
# MelonDMA (github.com/denmrnngp-cloud/MelonDMA): the DriverKit RDMA driver and its libibverbs-compat build
MELONDMA_DEXT ?=
ifeq ($(strip $(MELONDMA_DEXT)),)
$(error set MELONDMA_DEXT to MelonDMA's dext directory (usermode/libibverbs_compat/include and build/libibverbs))
endif
SIGN_ID ?= Apple Development

all: pd_rdma

pd_rdma: pd_rdma.c pd_rdma_common.h pd-rdma.entitlements
	clang $(CFLAGS) -I $(MELONDMA_DEXT)/usermode/libibverbs_compat/include $< \
	    -L $(MELONDMA_DEXT)/build -libverbs -Wl,-rpath,$(MELONDMA_DEXT)/build -o $@
	codesign --force --options runtime --timestamp=none --sign "$(SIGN_ID)" --entitlements pd-rdma.entitlements $@
	codesign --verify --strict $@
else
all: pd_rdma libpd_rdma_tx.so

pd_rdma: pd_rdma.c pd_rdma_common.h
	cc $(CFLAGS) $< -libverbs -o $@

libpd_rdma_tx.so: pd_rdma_tx.c pd_rdma_common.h
	cc $(CFLAGS) -fPIC -shared $< -libverbs -o $@
endif

clean:
	rm -f pd_rdma libpd_rdma_tx.so

.PHONY: all clean
