
//Defined only for charges along z direction
inline __host__ __device__ void GetFields(const complexd *plaqfield, int pos, int dirx, int diry, int dirz, bool evenradius, complexd field[6]) {	
	if(evenradius){
		//Ex
		complexd plaq = plaqfield[pos + dirx * Volume()];
		int s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + dirx * Volume()];
		field[0] = plaq * 0.5;
		//Ey
		plaq = plaqfield[pos + diry * Volume()];
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + diry * Volume()];
		field[1] = plaq * 0.5;
		//Ez
		plaq = plaqfield[pos + dirz * Volume()];
		s1 = indexNO_neg(pos, dirz, -1);
		plaq += plaqfield[s1 + dirz * Volume()];
		field[2] = plaq * 0.5;
		//Bx
		plaq = plaqfield[pos + (3 + dirx) * Volume()];
		s1 = indexNO_neg(pos, dirz, -1);
		plaq += plaqfield[s1 + (3 + dirx) * Volume()];
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + (3 + dirx) * Volume()];
		s1 = indexNO_neg(s1, dirz, -1);
		plaq += plaqfield[s1 + (3 + dirx) * Volume()];
		field[3] = plaq * 0.25;
		//By
		plaq = plaqfield[pos + (3 + diry) * Volume()];
		s1 = indexNO_neg(pos, dirz, -1);
		plaq += plaqfield[s1 + (3 + diry) * Volume()];
		s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + (3 + diry) * Volume()];
		s1 = indexNO_neg(s1, dirz, -1);
		plaq += plaqfield[s1 + (3 + diry) * Volume()];
		field[4] = plaq * 0.25;
		//Bz
		plaq = plaqfield[pos + (3 + dirz) * Volume()];
		s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		field[5] = plaq * 0.25;
	}
	else{
		//Valid for mid and charge plane
		//Valid only for odd radius
		//Ex
		complexd plaq = plaqfield[pos + dirx * Volume()];
		int s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + dirx * Volume()];
		s1 = indexNO_neg(pos, dirz, 1);
		plaq += plaqfield[s1 + dirx * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + dirx * Volume()];
		field[0] = plaq * 0.25;
		//Ey
		plaq = plaqfield[pos + diry * Volume()];
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + diry * Volume()];
		s1 = indexNO_neg(pos, dirz, 1);
		plaq += plaqfield[s1 + diry * Volume()];
		s1 = indexNO_neg(s1, diry, -1);
		plaq += plaqfield[s1 + diry * Volume()];
		field[1] = plaq * 0.25;
		//Ez
		plaq = plaqfield[pos + dirz * Volume()];
		field[2] = plaq;
		//Bx
		plaq = plaqfield[pos + (3 + dirx) * Volume()];
		s1 = indexNO_neg(s1, diry, -1);
		plaq += plaqfield[s1 + (3 + dirx) * Volume()];
		field[3] = plaq * 0.5;
		//By
		plaq = plaqfield[pos + (3 + diry) * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + (3 + diry) * Volume()];
		field[4] = plaq * 0.5;
		//Bz
		plaq = plaqfield[pos + (3 + dirz) * Volume()]; 
		s1 = indexNO_neg(pos, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()]; 
		s1 = indexNO_neg(pos, diry, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()]; 
		int s2 = indexNO_neg(pos, dirz, 1);
		plaq += plaqfield[s2 + (3 + dirz) * Volume()]; 
		s1 = indexNO_neg(s2, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(s2, diry, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		s1 = indexNO_neg(s1, dirx, -1);
		plaq += plaqfield[s1 + (3 + dirz) * Volume()];
		field[5] = plaq * 0.125;
	}
}








template<class Real>
__global__ void kernel_plaquette_comps(const Real *lat, complexd* plaq_comps, complexd* mean_plaq){
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;   
	complexd plaq[6];
	for(int d=0; d<6; d++) plaq[d] = 0.0;			
	if( idx < Volume() ) {
	   	uint id = idx;
		uint parity = 0;
		if( id >= HalfVolume() ){
			parity = 1;	
			id -= HalfVolume();
		}
		SixPlaquette(lat, plaq, id, parity);	
		int x[4];
		indexEO(id, parity, x);
		int pos = indexId(x);
		for(int d=0; d<6; d++) plaq_comps[pos + d * Volume()] = plaq[d];
	}
	for(int d=0; d<6; d++){
		reduce_block_1d<complexd>(mean_plaq + d, plaq[d]);
	  __syncthreads();
	}
}

	

template<class Real>
class PlaqFields: Tunable{
public:
	Array<complexd>* fields;
	Array<complexd>* Meanfields;
	Array<complexd>* Meanfields_dev;
private:
	Array<Real>* lat;
	int size;
	int sum;
	double timesec;
#ifdef TIMMINGS
    Timer time;
#endif

   unsigned int sharedBytesPerThread() const { return sizeof(complexd); }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
   bool tuneSharedBytes() const { return false; } // Don't tune shared memory
   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   unsigned int minThreads() const { return size; }
   void apply(const cudaStream_t &stream){
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	kernel_plaquette_comps<Real><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(lat->getPtr(), fields->getPtr(), Meanfields_dev->getPtr());
}
public:
   PlaqFields(Array<Real>* lat) : lat(lat) {
    size = Volume();
   	fields = new Array<complexd>(Device, 6*size );
   	Meanfields_dev = new Array<complexd>(Device, 6 );
   	Meanfields_dev->Clear();
   	Meanfields = new Array<complexd>(Host, 6 );
	timesec = 0.0;  
	sum = 0;
}
   ~PlaqFields(){ delete fields; delete Meanfields_dev; delete Meanfields; };
   Array<complexd>* Run(const cudaStream_t &stream){
#ifdef TIMMINGS
    time.start();
#endif
	apply(stream);
	sum++;
    cudaDevSync();
    cudaCheckError("Kernel execution failed");
#ifdef TIMMINGS
	cudaDevSync( );
    time.stop();
    timesec = time.getElapsedTimeInSec();
#endif
	return fields;
}
   Array<complexd>* Run(){ return Run(0); }
   
   Array<complexd>* getPlaqField(){ return fields; }
   
   Array<complexd>* GetMean(){
		Meanfields->Copy(Meanfields_dev);
		for(int i = 0; i < Meanfields->Size(); i++)
			Meanfields->at(i) /= double(size*sum);	   	
	   return Meanfields;
   }
   
   double flops(){	return ((double)flop() * 1.0e-9) / timesec;}
   double bandwidth(){	return (double)bytes() / (timesec * (double)(1 << 30));}
   long long flop() const { return 0;}
   long long bytes() const{ return 0;}
   double get_time(){	return timesec;}
   void stat(){	cout << "PlaqFields:  " <<  get_time() << " s\t"  << bandwidth() << " GB/s\t" << flops() << " GFlops"  << endl;}
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
  void preTune() { Meanfields_dev->Backup(); }
  void postTune() { Meanfields_dev->Restore(); }

};





__global__ void kernel_l2_multilevel_00(double *lat, complexd *poly, complexd *latmhit, bool ppmhit, bool plaqmhit){
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
	
	
	if(!ppmhit) poly[id4d] = exp_ir(lat[ indexId(x, TDir()) ]);
	else{
		if(plaqmhit){
			for(int dir = 0; dir < Dirs(); dir++){
				complexd res = MultiHit(lat, id, parity, dir);
				latmhit[id + parity * HalfVolume() + dir * Volume()] = res;
				if(dir == TDir() && ppmhit) poly[id4d] = res;//MultiHit(lat, id, parity, TDir());
			}		
		}
		else{
			if(ppmhit) poly[id4d] = MultiHit(lat, id, parity, TDir());
		}
	}
}


class Polyakov_Volume0: Tunable{
private:
	Array<double>* lat;
	Array<complexd>* poly;
	Array<complexd>* mhit;
	int size;
	double timesec;
	bool ppmhit;
	bool plaqmhit;
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
	if(plaqmhit) kernel_l2_multilevel_00<<<tp.grid, tp.block, 0, stream>>>(lat->getPtr(), poly->getPtr(), mhit->getPtr(), ppmhit, plaqmhit);
	else kernel_l2_multilevel_00<<<tp.grid, tp.block, 0, stream>>>(lat->getPtr(), poly->getPtr(), 0, ppmhit, plaqmhit);
}
public:
   Polyakov_Volume0(Array<double>* lat, bool ppmhit, bool plaqmhit) : lat(lat), ppmhit(ppmhit), plaqmhit(plaqmhit) {
	size = Volume();
	mhit = 0;
	poly = new Array<complexd>(Device, Volume());
	if(plaqmhit) mhit = new Array<complexd>(Device, Dirs()*Volume());
	timesec = 0.0;  
}
   ~Polyakov_Volume0(){ };
   Array<complexd>* GetLatMHit(){ return mhit; }
   Array<complexd>* GetPolyVol(){ return poly; }
   void Run(const cudaStream_t &stream){
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
}
   void Run(){	return Run(0);}
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







