#ifndef __MULTILEVEL__
#define __MULTILEVEL__

#include <tuple>
#include "random.h"
#include "array.h"
#include "parameters.h"

namespace U1{




class ML_Fields{
	public:
	Array<complexd>* plaq = 0;
	Array<complexd>* pp = 0;
	Array<complexd>* ppo = 0;
	
	ML_Fields(){}
	ML_Fields(Array<complexd>* plaq, Array<complexd>* pp, Array<complexd>* ppo) : plaq(plaq), pp(pp), ppo(ppo) {}
	void Set(Array<complexd>* plaq_, Array<complexd>* pp_, Array<complexd>* ppo_){
		plaq = plaq_;
		pp = pp_;
		ppo = ppo_;
	}
	~ML_Fields(){ 
		if(plaq) delete plaq;
		if(pp) delete pp;
		if(ppo) delete ppo;
	}
};




class MLArg{
	uint nl0 = 2;
	uint nl1 = 4;
	uint n4 = 1;
	uint k4 = 5;
	uint n2 = 10;
	uint k2 = 5;
	uint metrop = 1;
	uint ovrn = 3;
	uint rmax = 1;
	
	bool mhit = true;
	
	public:
	MLArg(){
		rmax = Grid(0)/2;
	};
	
	uint nLinksLvl0() const { return nl0; }
	uint& nLinksLvl0() { return nl0; }
	uint StepsLvl0() const { return n2; }
	uint& StepsLvl0() { return n2; }
	uint UpdatesLvl0() const { return k2; }
	uint& UpdatesLvl0() { return k2; }
	
	uint nLinksLvl1() const { return nl1; }
	uint& nLinksLvl1() { return nl1; }	
	uint StepsLvl1() const { return n4; }
	uint& StepsLvl1() { return n4; }
	uint UpdatesLvl1() const { return k4; }
	uint& UpdatesLvl1() { return k4; }
	
	
	uint nUpdatesMetropolis() const { return metrop; }
	uint& nUpdatesMetropolis() { return metrop; }
	
	uint nUpdatesOvr() const { return ovrn; }
	uint& nUpdatesOvr() { return ovrn; }
	
	
	uint Rmax() const { return rmax; }
	uint& Rmax() { return rmax; }
	
	bool MHit() const { return mhit; }
	bool& MHit() { return mhit; }
	
	void print(){
		using namespace std;
		cout << "==============================================" << endl;
		cout << "Rmax: " << rmax << endl;
		cout << "----------------------------------------------" << endl;
		cout << "Level 0:" << endl;
		cout << "\tNº time links per slice: " << nl0 << endl;
		cout << "\tNº iterations: " << n2 << endl;
		cout << "\tNº updates: " << k2 << endl;
		cout << "\tNº metropolis updates: " << metrop << endl;
		cout << "\tNº overrelaxation updates: " << ovrn << endl;
		cout << "----------------------------------------------" << endl;	
		cout << "Level 1:" << endl;
		cout << "\tNº time links per slice: " << nl1 << endl;
		cout << "\tNº iterations: " << n4 << endl;
		cout << "\tNº updates: " << k4 << endl;
		cout << "\tNº metropolis updates: " << metrop << endl;
		cout << "\tNº overrelaxation updates: " << ovrn << endl;
		cout << "----------------------------------------------" << endl;
		if(mhit) cout << "Using multihit." << endl;
		else cout << "Not using multihit." << endl;
		cout << "==============================================" << endl;	
	}
	
	void check(){
		using namespace std;
		bool cannot_run = false;
		
		
		if(Dirs() < 4){
			cout << "Error: Only implemented for 4D lattice..." << endl;
			cannot_run = true;
		}
		
		
		if( Grid(TDir())%nl0 != 0  ) {
			cannot_run = true;
			cout << "Error: Cannot Apply MultiLevel Algorithm..." << endl;
			cout << "\t\tNumber of time links are not multiple of " << nl0 << endl;
		
		}
		if( Grid(TDir())%nl1 != 0 ){
			cannot_run = true;
			cout << "Error: Cannot Apply MultiLevel Algorithm..." << endl;
			cout << "\t\tNumber of time links are not multiple of " << nl1 << endl;
		
		}
		if( (nl1%nl0) != 0 ) {
			cannot_run = true;
			cout << "Error: Cannot Apply MultiLevel Algorithm..." << endl;
			cout << "\t\tThe number of time links in level 1 (" << nl1 << ") is not multiple of the level 0 (" << nl0 << ")" << endl;
		}	
		if(rmax < 1 || rmax > Grid(0)/2+1){
			cout << "Error: Rmax should be >= 2 and <= " << Grid(0)/2+1 << endl;
			cannot_run = true;
		}

		if(cannot_run) exit(1);
	}
};












class MLTTOArg{
	uint nl0 = 2;
	uint nl1 = 4;
	uint n4 = 1;
	uint k4 = 5;
	uint n2 = 10;
	uint k2 = 5;
	uint metrop = 1;
	uint ovrn = 3;
	uint radius = 2;
	int2 perpPoint = make_int2(0, 0);
	
	bool squaredField = true;
	bool alongCharges = true;
	bool symmetrize = false;
	
	bool ppmhit = false;
	bool plaqmhit = false;
	
	public:
	MLTTOArg(){ };
	
	bool SquaredField() const { return squaredField; }
	bool& SquaredField() { return squaredField; }
	
	bool AlongCharges() const { return alongCharges; }
	bool& AlongCharges() { return alongCharges; }
	
	bool Sym() const { return symmetrize; }
	bool& Sym() { return symmetrize; }
		
	bool PPMHit() const { return ppmhit; }
	bool& PPMHit() { return ppmhit; }
	bool PlaqMHit() const { return plaqmhit; }
	bool& PlaqMHit() { return plaqmhit; }
	
	uint Radius() const { return radius; }
	uint& Radius() { return radius; }
	
	int2 PerpPoint() const { return perpPoint; }
	int2& PerpPoint(){ return perpPoint; }
	
	uint nLinksLvl0() const { return nl0; }
	uint& nLinksLvl0() { return nl0; }
	uint StepsLvl0() const { return n2; }
	uint& StepsLvl0() { return n2; }
	uint UpdatesLvl0() const { return k2; }
	uint& UpdatesLvl0() { return k2; }
	
	uint nLinksLvl1() const { return nl1; }
	uint& nLinksLvl1() { return nl1; }	
	uint StepsLvl1() const { return n4; }
	uint& StepsLvl1() { return n4; }
	uint UpdatesLvl1() const { return k4; }
	uint& UpdatesLvl1() { return k4; }
	
	
	uint nUpdatesMetropolis() const { return metrop; }
	uint& nUpdatesMetropolis() { return metrop; }
	
	uint nUpdatesOvr() const { return ovrn; }
	uint& nUpdatesOvr() { return ovrn; }
	
	void print(){
		using namespace std;
		cout << "==============================================" << endl;
		cout << "R: " << radius << endl;
		cout << "Charges at z direction, (-R/2, R/2)"<< endl;
		if(alongCharges) cout << "Results at (" << perpPoint.x << ", " << perpPoint.y << ", z)." << endl;
		else cout << "Results at (x, " << perpPoint.x << ", " << perpPoint.y << ")." << endl;
		cout << "----------------------------------------------" << endl;	
		if(squaredField) cout << "Squared Fields." << endl;
		else cout << "Non-squared Fields." << endl;
		if(alongCharges) cout << "Results along the charges." << endl;
		else cout << "Results perpendicular of the charges at the middle of the charges." << endl;
		if(symmetrize) cout << "Symmetrizing the results." << endl;
		cout << "----------------------------------------------" << endl;
		if(ppmhit) cout << "Applying MultiHit to the Polyakov lines." << endl;
		else cout << "Not applying MultiHit to the Polyakov lines." << endl;
		if(plaqmhit) cout << "Applying MultiHit to the Plaquettes." << endl;
		else cout << "Not applying MultiHit to the Plaquettes." << endl;
		cout << "----------------------------------------------" << endl;
		cout << "Level 0:" << endl;
		cout << "\tNº time links per slice: " << nl0 << endl;
		cout << "\tNº iterations: " << n2 << endl;
		cout << "\tNº updates: " << k2 << endl;
		cout << "\tNº metropolis updates: " << metrop << endl;
		cout << "\tNº overrelaxation updates: " << ovrn << endl;
		cout << "----------------------------------------------" << endl;	
		cout << "Level 1:" << endl;
		cout << "\tNº time links per slice: " << nl1 << endl;
		cout << "\tNº iterations: " << n4 << endl;
		cout << "\tNº updates: " << k4 << endl;
		cout << "\tNº metropolis updates: " << metrop << endl;
		cout << "\tNº overrelaxation updates: " << ovrn << endl;
		cout << "----------------------------------------------" << endl;	
	}
	
	void check(){
		using namespace std;
		bool cannot_run = false;
		
		
		if(Dirs() < 4){
			cout << "Error: Only implemented for 4D lattice..." << endl;
			cannot_run = true;
		}
		
		
		if( Grid(TDir())%nl0 != 0  ) {
			cannot_run = true;
			cout << "Error: Cannot Apply MultiLevel Algorithm..." << endl;
			cout << "\t\tNumber of time links are not multiple of " << nl0 << endl;
		
		}
		if( Grid(TDir())%nl1 != 0 ){
			cannot_run = true;
			cout << "Error: Cannot Apply MultiLevel Algorithm..." << endl;
			cout << "\t\tNumber of time links are not multiple of " << nl1 << endl;
		
		}
		if( (nl1%nl0) != 0 ) {
			cannot_run = true;
			cout << "Error: Cannot Apply MultiLevel Algorithm..." << endl;
			cout << "\t\tThe number of time links in level 1 (" << nl1 << ") is not multiple of the level 0 (" << nl0 << ")" << endl;
		}	
		if(radius < 2 || radius > Grid(0)/2+1){
			cout << "Error: Radius should be >= 2 and <= " << Grid(0)/2+1 << endl;
			cannot_run = true;
		}
		
		
		if(perpPoint.x > Grid(0)/2-1 || perpPoint.x < -Grid(0)/2 || perpPoint.y > Grid(0)/2-1 || perpPoint.y < -Grid(0)/2){
			cout << "Perpendicular points (" << perpPoint.x << ", " << perpPoint.y <<") should be between [" << -Grid(0)/2 << ":" << Grid(0)/2-1 << "]" << endl;
			cannot_run = true;
		}
		if(cannot_run) exit(1);
	}
};


template<bool multihit>
std::vector<double> MultiLevel(Array<double> *lat, CudaRNG *rng_state, MLArg *arg, bool PrintResultsAtEveryN4);


Array<complexd>* MultiLevel(Array<double> *lat, CudaRNG *rng_state, MLArg *arg, bool PrintResultsAtEveryN4=false);

std::tuple<Array<complexd>*, Array<complexd>*> MultiLevelField(Array<double> *lat, CudaRNG *rng_state, MLArg *arg, bool PrintResultsAtEveryN4=false);


ML_Fields* MultiLevelTTO(Array<double> *lat, CudaRNG *rng_state, MLTTOArg *arg);


}
#endif
