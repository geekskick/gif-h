#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "gif.h"

int main(){
	struct gif_writer writer;
	const unsigned int w = 100;
	const unsigned int h = 100;
	const unsigned int img_sz = w * h * 4 * sizeof(uint8_t); //RGBA!

	
	bool rc = gif_begin(&writer, "TestGif.gif", w, h, 100);
	if(!rc){
		fprintf(stderr, "Unsuccessful GifBegin\n");
		return 0;
	}
	
	uint8_t img[img_sz];
	memset(img, 0, img_sz);
	gif_write_frame(&writer, img, w, h, 100, 1, false);
	memset(img, 0xff, img_sz);
	gif_write_frame(&writer, img, w, h, 100, 1, false);
	memset(img, 0, img_sz);
	gif_write_frame(&writer, img, w, h, 100, 1, false);
	memset(img, 0xff, img_sz);
	gif_write_frame(&writer, img, w, h, 100, 1, false);

	
	rc = gif_end(&writer);
	if(!rc){
		fprintf(stderr, "Unsuccessful GifEnd\n");
		return 0;
	}
	return 0;
}