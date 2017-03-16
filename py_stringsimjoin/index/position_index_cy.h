#include <map>
#include <utility>
#include <vector>
#include <math.h>

class PositionIndexCy {
  public:
    std::map<int, std::vector< std::pair<int, int> > > index;                       
    int min_len, max_len;                                          
    std::vector<int> size_vector, l_empty_ids;                                       
    double threshold;

    PositionIndexCy();
    PositionIndexCy(std::map<int, std::vector< std::pair<int, int> > >&, std::vector<int>&, std::vector<int>&, int&, int&, double&);
    ~PositionIndexCy();
    void set_fields(std::map<int, std::vector< std::pair<int, int> > >&, std::vector<int>&, std::vector<int>&, int&, int&, double&);  
};
