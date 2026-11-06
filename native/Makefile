# libnailong.so：解码 + 缩放/HSV + 特征/评分 + 线程池，结果与 Python 参考实现逐位一致。
# 需要：x86-64（AVX2/FMA），gcc，libjpeg-turbo 3.2.x（含 jpegint.h 头文件），libdeflate，libwebp。
# 必须在运行它的机器上编译（-march=native），且不要加 -ffast-math。
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -ffp-contract=off -fno-fast-math -fno-math-errno -fPIC -pthread -Wall -Wno-unused-function
LIBS    ?= -ljpeg -ldeflate -lwebp -lwebpdemux -lm
SRCS     = src/driver.c src/decode.c src/prep.c src/feat.c

libnailong.so: $(SRCS) include/nl.h src/jhuff.h src/prep_internal.h
	$(CC) $(CFLAGS) -Iinclude -Isrc -shared -o $@ $(SRCS) $(LIBS)

clean:
	rm -f libnailong.so
