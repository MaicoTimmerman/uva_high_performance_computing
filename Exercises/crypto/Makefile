debug		:= 0
include		./Makefile.inc


all: clean encrypt

encrypt: encrypt.cu
	$(NVCC) -c encrypt.cu $(NVCCFLAGS) $(INCLUDES)
	$(LINKER) -o $(PROJ_BASE)/encrypt encrypt.o timer.cc $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)


run: encrypt
	./encrypt


clean:
	rm -f ./encrypt.o
	rm -f $(PROJ_BASE)/encrypt
	rm -f cuda.data recovered.data sequential.data sequential_decrypted.data
