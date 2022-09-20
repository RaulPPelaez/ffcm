

#include "Integrator/BDHI/FCM/FCM_impl.cuh"
#include<thrust/device_vector.h>
using namespace uammd;
//Lets give the gpu container a name, in case we want to change it later.
template <class T>
using gpu_container = thrust::device_vector<T>;

// The FCM can be written as: dX/dt = S^T FFT^{-1} L FFT S F
// -S is the spreading operator
// -FFT is the fast fourier transform
// L is the Stokes solution operator in Fourier space
// F are the forces and/or torques acting on the particles
// X are the positions and/or orientations of the particles

// The spreading operator communicates a series of particles with a field
// discretized on a grid.
// This communication is mediated via a kernel, such that, for a given position
// in the grid:
// \vec{r} (S F) (\vec{r}) = \sum_i \delta_a(\vec{r} - \vec{q}_i) F_i
// Where i goes over the particles, located at \vec{q}_i.

// In this case the FCM is going to be used as a part of a bigger code, called
// F-FCM, which involves using the above formalism with a modified spreading
// kernel. In particular, we define:
// \delta_a(\vec{r}) = (a + b r^2 + c)\exp(\tau r^2)

// In UAMMD the kernel is expected to be separable, so that
// \delta_a(\vec{r}) F_i = SomeOperation(F_i, \phi(r_x), \phi(r_y), \phi(rz)).
// In order to encode our goal in UAMMD we will need two parts:
// 1- A kernel that computes phi as a pair of values, returning r_\alpha and
//    \exp(\tau r_\alpha^2) (Gaussian below)
// 2- An object that performs SomeOperation, in this case returning the above
//    expression for \delta multiplied by the force (WeightCompute below)



//A Gaussian kernel compatible with the IBM module.
class Gaussian{
  const real tau;
public:
  const int support;
  Gaussian(real tau, int support):
    tau(tau),
    support(support){}

  __host__ __device__ int3 getSupport(real3 pos, int3 cell){
    return {support, support, support};
  }

  __device__ real2 phi(real r, real3 pos) const{
    return {r, exp(tau*r*r)};
  }

};

// An object compatible with the uammd IBM module.
// In spreading mode, this class will be provided with the kernel and the
// quantity for a given particle and cell and will return the final contribution
// to a cell.
// In interpolation mode (gather) this class gets the value of a cell and the
// kernel and will return the contribution to a particle.
// Since spreading and interpolation are adjoint operations, the same function
// is used for both responsabilities.
struct WeightCompute{
  real a,b,c;
  WeightCompute(real a, real b, real c):
    a(a), b(b), c(c){}


  __device__ real getKernelWeight(thrust::tuple<real2, real2, real2> kernel){
    real rx = thrust::get<0>(kernel).x;
    real phiX = thrust::get<0>(kernel).y;
    real ry = thrust::get<1>(kernel).x;
    real phiY = thrust::get<1>(kernel).y;
    real rz = thrust::get<2>(kernel).x;
    real phiZ = thrust::get<2>(kernel).y;
    return (a+rx*rx+ry*ry+rz*rz*b + b)*phiX*phiY*phiZ;
  }

  template<class T>
  __device__ auto operator()(T value, thrust::tuple<real2, real2, real2> kernel){
    real k = getKernelWeight(kernel);
    return value*k;
  }
};


int main(){
  int numberParticles = 1;
  gpu_container<real4> pos(numberParticles);
  gpu_container<real4> force(numberParticles);
  gpu_container<real4> torque(numberParticles);
  //Fill everything with zeros
  thrust::fill(force.begin(), force.end(), real4());
  thrust::fill(torque.begin(), torque.end(), real4());
  thrust::fill(pos.begin(), pos.end(), real4());
  //A force on the first particle in the X direction
  force[0] = {1, 0, 0, 0};

  //Some arbitrary parameters for the kernel
  real a = 1;
  real b = 1;
  real c = 1;
  real tau = 1;
  int support = 5;
  // UAMMD requests most objects and functors as shared pointers.
  // Besides guarding the code against memory errors shared_ptr
  //  eases the sending of objects to the GPU.
  auto kernel = std::make_shared<Gaussian>(tau, support);
  auto wc = std::make_shared<WeightCompute>(a,b,c);

  //The UAMMD fcm module can be specialized for a certain kernel for the monopole and the dipoles,
  // and also for a certain WeightCompute for the IBM module.
  using FCM = uammd::BDHI::FCM_impl<Gaussian, Gaussian, WeightCompute>;
  FCM::Parameters par;
  par.cells = {32,32,32};   //Grid dimensions
  par.box = Box({32,32,32}); //Domain size
  par.kernel = kernel;       //The ibm kernels
  par.kernelTorque = kernel; //You probably want another different instance for torques.
  par.wc = wc;               //The weight compute for ibm
  par.viscosity = 1.0;       //An arbitrary viscosity
  auto fcm = std::make_shared<FCM>(par);


    // Computes the hydrodynamic displacements, defined as
    //d\vec{q} = M FT + pref\sqrt{2kT M} dW.
    // The positions, q, the forces, F and the torques, T, must be passed as pointers to real4 with interleaved x,y,z components for each marker. The fourth element is unused.
    // The return type is a pair containing two gpu containers. The first element holds the linear displacements, while the second holds the dipolar displacements.
  real kT = 0; //No fluctuations
  real pref = 0;
  //We take the addresses of the containers
  auto pos_ptr = pos.data().get();
  auto force_ptr = force.data().get();
  auto torque_ptr = torque.data().get();
  cudaStream_t st = 0; //Let us use the default stream
  auto displacements = fcm->computeHydrodynamicDisplacements(pos_ptr, force_ptr, torque_ptr,
							     numberParticles,
							     kT, pref,
							     st);

  auto monopoles = displacements.first;
  auto dipoles = displacements.second;
  //Lets download the results to the CPU for printing
  std::vector<real3> host_monopoles(numberParticles);
  //Thrust is smart enough to detect that this is a GPU to CPU copy
  thrust::copy(monopoles.begin(), monopoles.end(), host_monopoles.begin());

  //Lets print all of them
  for(auto mf: host_monopoles){
    std::cout<<mf<<std::endl;
  }
  return 0;
}
