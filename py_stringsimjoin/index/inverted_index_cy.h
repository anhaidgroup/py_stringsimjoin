#include <map>
#include <vector>

class InvertedIndexCy {
  public:
    std::map<int, std::vector<int> > index;                       
    std::vector<int> size_vector;                                       

    InvertedIndexCy();
    InvertedIndexCy(std::map<int, std::vector<int> >&, std::vector<int>&);
    ~InvertedIndexCy();
    void set_fields(std::map<int, std::vector<int> >&, std::vector<int>&);  
    void build_prefix_index(std::vector< std::vector<int> >&, int, double);
};
