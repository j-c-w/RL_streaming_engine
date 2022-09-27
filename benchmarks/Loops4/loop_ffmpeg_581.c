// Source is: /home/alex/.local/share/compy-Learn/1.0/ffmpeg/content/libswresample/rematrix.c

#include <stdint.h>
#include <stdio.h>




typedef int64_t integer;

int
fn (float *coeffp, int i, integer len, float **out, const float **in)
{
  for (i = 0; i < len; i++)
    {
      float t =
	in[2][i] * (float) coeffp[0 * 8 + 2] +
	in[3][i] * (float) coeffp[0 * 8 + 3];
      out[0][i] =
	t + in[0][i] * (float) coeffp[0 * 8 + 0] +
	in[4][i] * (float) coeffp[0 * 8 + 4] +
	in[6][i] * (float) coeffp[0 * 8 + 6];
      out[1][i] =
	t + in[1][i] * (float) coeffp[1 * 8 + 1] +
	in[5][i] * (float) coeffp[1 * 8 + 5] +
	in[7][i] * (float) coeffp[1 * 8 + 7];
}}
