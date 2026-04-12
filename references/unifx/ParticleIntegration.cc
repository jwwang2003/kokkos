#include <algorithm>
#include <iostream>
#include <random>

#include "Common/Macros.hh"
#include "Hal/Array.hh"
#include "Hal/HalContext.hh"
#include "Hal/Parallel.hh"

using namespace unifx;

struct ParticleSystem {
  hal::UnifXArray<float*> pos_x, pos_y, pos_z;
  hal::UnifXArray<float*> vel_x, vel_y, vel_z;
  hal::UnifXArray<float*> force_x, force_y, force_z;
  hal::UnifXArray<float*> inv_mass;

  explicit ParticleSystem(std::size_t n)
      : pos_x("pos_x", n),
        pos_y("pos_y", n),
        pos_z("pos_z", n),
        vel_x("vel_x", n),
        vel_y("vel_y", n),
        vel_z("vel_z", n),
        force_x("force_x", n),
        force_y("force_y", n),
        force_z("force_z", n),
        inv_mass("inv_mass", n) {}

  [[nodiscard]] std::size_t size() const { return pos_x.Size(); }
};

double InitializeParticles(ParticleSystem& ps, unsigned seed = 42) {
  Kokkos::Timer timer;
  const std::size_t n = ps.size();

  auto h_pos_x = hal::CreateMirrorView(ps.pos_x);
  auto h_pos_y = hal::CreateMirrorView(ps.pos_y);
  auto h_pos_z = hal::CreateMirrorView(ps.pos_z);

  auto h_vel_x = hal::CreateMirrorView(ps.vel_x);
  auto h_vel_y = hal::CreateMirrorView(ps.vel_y);
  auto h_vel_z = hal::CreateMirrorView(ps.vel_z);

  auto h_force_x = hal::CreateMirrorView(ps.force_x);
  auto h_force_y = hal::CreateMirrorView(ps.force_y);
  auto h_force_z = hal::CreateMirrorView(ps.force_z);

  auto h_inv_mass = hal::CreateMirrorView(ps.inv_mass);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
  std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> mass_dist(0.5f, 2.0f);

  for (std::size_t i = 0; i < n; ++i) {
    h_pos_x(i) = pos_dist(rng);
    h_pos_y(i) = 5.0f + pos_dist(rng) * 0.2f;
    h_pos_z(i) = pos_dist(rng);

    h_vel_x(i) = vel_dist(rng);
    h_vel_y(i) = vel_dist(rng);
    h_vel_z(i) = vel_dist(rng);

    h_force_x(i) = 0.0f;
    h_force_y(i) = -9.81f;
    h_force_z(i) = 0.0f;

    const float mass = mass_dist(rng);
    h_inv_mass(i)    = 1.0f / mass;
  }

  for (std::size_t i = 0; i < std::min<std::size_t>(16, n); ++i) {
    h_inv_mass(i) = 0.0f;
  }

  hal::DeepCopy(ps.pos_x, h_pos_x);
  hal::DeepCopy(ps.pos_y, h_pos_y);
  hal::DeepCopy(ps.pos_z, h_pos_z);

  hal::DeepCopy(ps.vel_x, h_vel_x);
  hal::DeepCopy(ps.vel_y, h_vel_y);
  hal::DeepCopy(ps.vel_z, h_vel_z);

  hal::DeepCopy(ps.force_x, h_force_x);
  hal::DeepCopy(ps.force_y, h_force_y);
  hal::DeepCopy(ps.force_z, h_force_z);

  hal::DeepCopy(ps.inv_mass, h_inv_mass);

  hal::Fence();

  return timer.seconds() * 1000.0;
}

void StepParticles(ParticleSystem& ps, float dt, float damping, float ground_y,
                   float restitution, float ground_friction) {
  auto pos_x = ps.pos_x;
  auto pos_y = ps.pos_y;
  auto pos_z = ps.pos_z;

  auto vel_x = ps.vel_x;
  auto vel_y = ps.vel_y;
  auto vel_z = ps.vel_z;

  auto force_x = ps.force_x;
  auto force_y = ps.force_y;
  auto force_z = ps.force_z;

  auto inv_mass = ps.inv_mass;

  const int n = static_cast<int>(ps.size());

  hal::ParallelFor(
      "step_particles", n, UNIFX_LAMBDA(const int i) {
        const float w = inv_mass(i);
        if (w == 0.0f) {
          return;
        }

        vel_x(i) += dt * force_x(i) * w;
        vel_y(i) += dt * force_y(i) * w;
        vel_z(i) += dt * force_z(i) * w;

        vel_x(i) *= damping;
        vel_y(i) *= damping;
        vel_z(i) *= damping;

        pos_x(i) += dt * vel_x(i);
        pos_y(i) += dt * vel_y(i);
        pos_z(i) += dt * vel_z(i);

        if (pos_y(i) < ground_y) {
          pos_y(i) = ground_y;

          if (vel_y(i) < 0.0f) {
            vel_y(i) = -vel_y(i) * restitution;
          }

          vel_x(i) *= ground_friction;
          vel_z(i) *= ground_friction;
        }
      });
}

double BenchmarkHotKernel(ParticleSystem& ps, int steps, float dt,
                          float damping, float ground_y, float restitution,
                          float ground_friction) {
  hal::Fence();
  Kokkos::Timer timer;

  for (int s = 0; s < steps; ++s) {
    StepParticles(ps, dt, damping, ground_y, restitution, ground_friction);
  }

  hal::Fence();
  return timer.seconds() * 1000.0;
}

void PrintSampleParticle(const ParticleSystem& ps, int idx) {
  auto h_pos_x = hal::CreateMirrorViewAndCopy(ps.pos_x);
  auto h_pos_y = hal::CreateMirrorViewAndCopy(ps.pos_y);
  auto h_pos_z = hal::CreateMirrorViewAndCopy(ps.pos_z);
  auto h_vel_x = hal::CreateMirrorViewAndCopy(ps.vel_x);
  auto h_vel_y = hal::CreateMirrorViewAndCopy(ps.vel_y);
  auto h_vel_z = hal::CreateMirrorViewAndCopy(ps.vel_z);

  std::cout << "Sample particle " << idx << ": " << "pos=(" << h_pos_x(idx)
            << ", " << h_pos_y(idx) << ", " << h_pos_z(idx) << "), " << "vel=("
            << h_vel_x(idx) << ", " << h_vel_y(idx) << ", " << h_vel_z(idx)
            << ")\n";
}

int main(int argc, char* argv[]) {
  hal::Initialize(argc, argv);
  {
    constexpr std::size_t n = 1'000'000;
    constexpr int steps     = 200;

    constexpr float dt              = 1.0f / 60.0f;
    constexpr float damping         = 0.999f;
    constexpr float ground_y        = 0.0f;
    constexpr float restitution     = 0.2f;
    constexpr float ground_friction = 0.95f;

    std::cout << "Particle count: " << n << "\n";
    std::cout << "Step count: " << steps << "\n";

    Kokkos::Timer total_timer;

    ParticleSystem ps(n);

    const double prepare_ms = InitializeParticles(ps);
    const double hot_ms = BenchmarkHotKernel(ps, steps, dt, damping, ground_y,
                                             restitution, ground_friction);

    const double total_ms = total_timer.seconds() * 1000.0;

    std::cout << "Prepare data time: " << prepare_ms << " ms\n";
    std::cout << "Hot kernel time : " << hot_ms << " ms\n";
    std::cout << "Per-step time   : " << (hot_ms / static_cast<double>(steps))
              << " ms\n";
    std::cout << "Total time      : " << total_ms << " ms\n";

    PrintSampleParticle(ps, 123);
  }
  hal::Finalize();
  return 0;
}