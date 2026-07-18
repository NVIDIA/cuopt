#include <cuopt/mathematical_optimization/io/lp_writer.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/io/data_model.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cuopt::mathematical_optimization::io;

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){ std::cout << "FAIL: " << msg << "\n"; ++failures; } } while(0)
static void near(double a, double b, const std::string& m){ if(std::fabs(a-b) > 1e-9 && !(std::isinf(a)&&std::isinf(b)&&((a>0)==(b>0)))){ std::cout<<"FAIL(near): "<<m<<" got "<<b<<" want "<<a<<"\n"; ++failures; } }
static std::unordered_map<std::string,int> idx(const std::vector<std::string>& n){ std::unordered_map<std::string,int> m; for(size_t i=0;i<n.size();++i)m[n[i]]=(int)i; return m; }

static void round(const std::string& tag, const std::string& lp){
  data_model_t<int,double> a = read_lp_from_string<int,double>(lp);
  std::string path = std::string("/home/nfs/iroy/lp_writer/.lp_out_")+tag+".lp";
  lp_writer_t<int,double> w(a);
  w.write(path);
  data_model_t<int,double> b = read_lp<int,double>(path);
  CHECK(a.get_sense()==b.get_sense(), tag+" sense");
  near(a.get_objective_offset(), b.get_objective_offset(), tag+" offset");
  const auto& an=a.get_variable_names(); const auto& bn=b.get_variable_names();
  CHECK(an.size()==bn.size(), tag+" nvars");
  auto bi=idx(bn);
  const auto& ac=a.get_objective_coefficients(); const auto& bc=b.get_objective_coefficients();
  const auto& alb=a.get_variable_lower_bounds(); const auto& blb=b.get_variable_lower_bounds();
  const auto& aub=a.get_variable_upper_bounds(); const auto& bub=b.get_variable_upper_bounds();
  const auto& at=a.get_variable_types(); const auto& bt=b.get_variable_types();
  for(size_t i=0;i<an.size();++i){ if(!bi.count(an[i])){ std::cout<<"FAIL missing var "<<an[i]<<"\n"; ++failures; continue;} int j=bi[an[i]];
    near(ac[i],bc[j],tag+" c "+an[i]); near(alb[i],blb[j],tag+" lb "+an[i]); near(aub[i],bub[j],tag+" ub "+an[i]);
    CHECK(at[i]==bt[j], tag+" type "+an[i]); }
  CHECK(a.get_row_names().size()==b.get_row_names().size(), tag+" nrows");
  CHECK(a.has_quadratic_objective()==b.has_quadratic_objective(), tag+" qobj");
  CHECK(a.get_quadratic_constraints().size()==b.get_quadratic_constraints().size(), tag+" nqc");
  std::remove(path.c_str());
  std::cout << tag << " done\n";
}

int main(){
  round("simple", "Minimize\n obj: 3 x + 2 y - z\nSubject To\n c1: x + y <= 10\n c2: x - z >= -4\n c3: 2 x + y = 6\nBounds\n 0 <= x <= 8\n y >= 1\n -5 <= z <= 5\nEnd\n");
  round("mip", "Minimize\n obj: x + y + 2 z\nSubject To\n c1: x + y + z <= 5\nBounds\n 0 <= y <= 10\nGenerals\n y\nBinaries\n x\n z\nEnd\n");
  round("qpobj", "Minimize\n obj: x + [ 2 x ^ 2 + 4 x * y + 6 y ^ 2 ] / 2\nSubject To\n c1: x + y >= 1\nEnd\n");
  round("qcqp", "Minimize\n obj: x + y\nSubject To\n lin: x + y <= 10\n qc: x + [ x ^ 2 + y ^ 2 ] <= 4\nEnd\n");
  std::cout << "\nTOTAL FAILURES: " << failures << "\n";
  return failures?1:0;
}
