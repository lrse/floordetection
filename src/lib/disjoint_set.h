
#ifndef __DISJOINT_SET__
#define __DISJOINT_SET__

// disjoint-set forests using union-by-rank and path compression (sort of).

typedef enum {GROUND, NON_GROUND} type;

typedef struct {
  int rank;
  int p;
  int size;
  type type_of;
  int total_pixels;
  int marked_pixels;    
} uni_elt;

class universe {
public:
  universe(int elements);
  ~universe();
  int find(int x);  
  void join(int x, int y);
  int size(int x) const { return elts[x].size; }
  int num_sets() const { return num; }
  uni_elt* get_elt(int x) { return (uni_elt*) &elts[x];}  

private:
  uni_elt *elts;
  int num;
};

#endif
