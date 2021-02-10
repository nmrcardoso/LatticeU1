
__global__ void kernel_metropolis_multilevel(double *lat, int parity, int mu, cuRNGState *rng_state, int nl){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
    
	int x[4];
	indexEO(id, parity, x);
	if((x[TDir()]%nl) || mu==3){	
		cuRNGState localState = rng_state[ id ];
		double new_phase = Random<double>(localState) * 2. * M_PI;
		double b = Random<double>(localState);
		rng_state[ id ] = localState;
    	double dS = MetropolisFunc(lat, id, parity, mu, new_phase);
		if(dS > b){
			lat[id + parity * HalfVolume() + mu * Volume()] = new_phase;
		}	
	}
}



class Metropolis_ML: Tunable{
private:
	Array<double>* lat;
	CudaRNG *rng_state;
	int nl;
	int metrop;
	int parity;
	int mu;
	int size;
	double timesec;
#ifdef TIMMINGS
    Timer time;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	kernel_metropolis_multilevel<<<tp.grid, tp.block, 0, stream>>>(lat->getPtr(), parity, mu, rng_state->getPtr(), nl);
}
public:
   Metropolis_ML(Array<double>* lat, CudaRNG *rng_state) : lat(lat), rng_state(rng_state){
	size = HalfVolume();
	timesec = 0.0;  
}
   ~Metropolis_ML(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	for(int m = 0; m < metrop; ++m)
	for(parity = 0; parity < 2; ++parity)
	for(mu = 0; mu < Dirs(); ++mu)
	    apply(stream);
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
}
   void Run(int metrop_, int nl_){ 
   	metrop = metrop_; 
   	nl = nl_;	
   	Run(0);
   }
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "Metropolis_ML:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size;
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() {
	lat->Backup();
	rng_state->Backup();
  }
  void postTune() {  
  	lat->Restore();
  	rng_state->Restore();
 }

};


__global__ void kernel_overrelaxation_multilevel(double *lat, int parity, int mu, int nl){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if( id >= HalfVolume() ) return ;
    
	int x[4];
	indexEO(id, parity, x);
	if((x[TDir()]%nl) || mu==3){		
		lat[id + parity * HalfVolume() + mu * Volume()] = OvrFunc(lat, id, parity, mu);
	}
}


class OverRelaxation_ML: Tunable{
private:
	Array<double>* lat;
	int nl;
	int ovrn;
	int parity;
	int mu;
	int size;
	double timesec;
#ifdef TIMMINGS
    Timer time;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	kernel_overrelaxation_multilevel<<<tp.grid, tp.block, 0, stream>>>(lat->getPtr(), parity, mu, nl);
}
public:
   OverRelaxation_ML(Array<double>* lat) : lat(lat) {
	size = HalfVolume();
	timesec = 0.0;  
}
   ~OverRelaxation_ML(){};
   void Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	for(int m = 0; m < ovrn; ++m)
	for(parity = 0; parity < 2; ++parity)
	for(mu = 0; mu < Dirs(); ++mu){
		//cout << multilevel << '\t' << mu << '\t' << parity << '\t' << m << endl;
	    apply(stream);
    }
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
}
   void Run(int ovrn_, int nl_){ 
   	ovrn = ovrn_; 
   	nl = nl_;	
   	Run(0);
   }
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "OverRelaxation_ML:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size;
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() {
  	lat->Backup();		
  }
  void postTune() {  
	lat->Restore();
 }

};
















template<bool multihit>
__global__ void kernel_l2_multilevel_0(double *lat, complexd *poly){
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;   
	if( id >= Volume() ) return;
	int parity = 0;
	if( id >= Volume()/2 ){
		parity = 1;	
		id -= Volume()/2;
	}	
	int x[4];
	indexEO(id, parity, x);
	int id4d = indexId(x);
	
	if(multihit){	
		poly[id4d] = MultiHit(lat, id, parity, TDir());
	}
	else{
		poly[id4d] = exp_ir(lat[ indexId(x, TDir()) ]);
	}
}


template< bool multihit>
class Polyakov_Volume: Tunable{
private:
	Array<double>* lat;
	Array<complexd>* poly;
	int size;
	double timesec;
#ifdef TIMMINGS
    Timer time;
#endif

   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	kernel_l2_multilevel_0<multihit><<<tp.grid, tp.block, 0, stream>>>(lat->getPtr(), poly->getPtr());
}
public:
   Polyakov_Volume(Array<double>* lat) : lat(lat) {
	size = Volume();
	poly = new Array<complexd>(Device, Volume());
	timesec = 0.0;  
}
   ~Polyakov_Volume(){ };
   Array<complexd>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return poly;
}
   Array<complexd>* Run(){	return Run(0);}
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "OverRelaxation:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
  TuneKey tuneKey() const {
    std::stringstream vol, aux;
    vol << PARAMS::Grid[0] << "x";
    vol << PARAMS::Grid[1] << "x";
    vol << PARAMS::Grid[2] << "x";
    vol << PARAMS::Grid[3];
    aux << "threads=" << size;
    return TuneKey(vol.str().c_str(), typeid(*this).name(), aux.str().c_str());
  }
  std::string paramString(const TuneParam &param) const {
    std::stringstream ps;
    ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << ")";
    ps << "shared=" << param.shared_bytes;
    return ps.str();
  }
  void preTune() { }
  void postTune() { }

};
