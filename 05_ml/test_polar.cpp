#include <cstdio>
#include <cstdlib>
#include <vector>
#include "nn_polar.h"
static void* al(size_t n){ return malloc(n); }
// 引数 t: 2枚目以降は追跡モード(前回のずれの近くだけ探す)
int main(int argc,char**argv){ static PolarWork W; if(!polarInit(W,al)) return 1; bool track=argc>1;
  std::vector<uint8_t> im(640*480); int i=0;
  while(fread(im.data(),1,im.size(),stdin)==im.size()){ PolarResult r=polarRead(W,im.data(),640,480,!(track&&i>0)); i++; printf("%.6f %.6f %d %d %.4f\n",r.value,r.conf,r.dx,r.dy,r.ncc);} }
