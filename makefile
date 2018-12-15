CC = nvcc
CFLAGS = -lcublas -lcurand -lcusolver  -Wno-deprecated-gpu-targets 

a.out: common.o paragibbs.o dist_gamma.o main.o 
	$(CC) -o $@ $^ $(CFLAGS)
	clear
	@echo "\nSUCCESSFUL COMPILATION!\n"

dist_gamma.o: dist_gamma.cu common.h 
	$(CC) -o $@ -c $< $(CFLAGS)

main.o: main.cu common.h paragibbs.h
	$(CC) -o $@ -c $< $(CFLAGS)

paragibbs.o: paragibbs.cu paragibbs.h common.h
	$(CC) -o $@ -c $< $(CFLAGS)

common.o: common.cu common.h
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm a.out *.o