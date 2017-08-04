#include "position_index_cy.h"

PositionIndexCy::PositionIndexCy(std::map<int, std::vector< std::pair<int, int> > >& ind, std::vector<int>& sv, std::vector<int>& emp_ids, int& min_l, int& max_l, double& t) {
  index = ind;
  size_vector = sv;
  l_empty_ids = emp_ids;
  min_len = min_l;
  max_len = max_l;
  threshold = t;
}

void PositionIndexCy::set_fields(std::map<int, std::vector< std::pair<int, int> > >& ind, std::vector<int>& sv, std::vector<int>& emp_ids, int& min_l, int& max_l, double& t) {
  index = ind;                                                                  
  size_vector = sv;                                                             
  l_empty_ids = emp_ids;
  min_len = min_l;                                                              
  max_len = max_l;                                                              
  threshold = t;                                                                
}

PositionIndexCy::PositionIndexCy() {}

PositionIndexCy::~PositionIndexCy() {} 
