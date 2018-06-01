#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "gif.h"

int main(){
	struct GifWriter writer;
	const unsigned int w = 100;
	const unsigned int h = 100;
	const unsigned int img_sz = w * h * 4 * sizeof(uint8_t); //RGBA!

	
	bool rc = GifBegin(&writer, "TestGif.gif", w, h, 100, 8, false);
	if(!rc){
		fprintf(stderr, "Unsuccessful GifBegin\n");
		return 0;
	}
	
	uint8_t img[img_sz];
	memset(img, 0, img_sz);
	GifWriteFrame(&writer, img, w, h, 100, 1, false);
	memset(img, 0xff, img_sz);
	GifWriteFrame(&writer, img, w, h, 100, 1, false);
	memset(img, 0, img_sz);
	GifWriteFrame(&writer, img, w, h, 100, 1, false);
	memset(img, 0xff, img_sz);
	GifWriteFrame(&writer, img, w, h, 100, 1, false);

	
	rc = GifEnd(&writer);
	if(!rc){
		fprintf(stderr, "Unsuccessful GifEnd\n");
		return 0;
	}
	return 0;
}