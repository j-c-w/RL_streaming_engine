// Source is: /home/alex/.local/share/compy-Learn/1.0/freeimage/content/Source/LibWebP/src/enc/iterator_enc.c

#include <stdint.h>
#include <stdio.h>




typedef __uint8_t uint8_t;

int
fn (uint8_t * const top, int i, const uint8_t * const blk)
{
  for (i = 0; i <= 2; ++i)
    {
      top[i] = blk[3 + (2 - i) * 32];
    }
}
