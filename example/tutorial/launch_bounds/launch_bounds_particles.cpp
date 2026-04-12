// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>

using exec_space = Kokkos::DefaultExecutionSpace;

struct ParticleSystem {
  using view_type = Kokkos::View<float*>;

  view_type pos_x;
  view_type pos_y;
  view_type pos_z;
  view_type vel_x;
  view_type vel_y;
  view_type vel_z;
  view_type force_x;
  view_type force_y;
  view_type force_z;
  view_type inv_mass;

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

  [[nodiscard]] std::size_t size() const { return pos_x.extent(0); }
};

struct SimulationParams {
  float dt;
  float damping;
  float ground_y;
  float restitution;
  float ground_friction;
};

struct SimulationMetrics {
  float average_height;
  float total_kinetic_energy;
  float max_speed;
  long long active_particles;
};

double initialize_particles(ParticleSystem& ps, unsigned seed = 42) {
  Kokkos::Timer timer;
  auto h_pos_x    = Kokkos::create_mirror_view(ps.pos_x);
  auto h_pos_y    = Kokkos::create_mirror_view(ps.pos_y);
  auto h_pos_z    = Kokkos::create_mirror_view(ps.pos_z);
  auto h_vel_x    = Kokkos::create_mirror_view(ps.vel_x);
  auto h_vel_y    = Kokkos::create_mirror_view(ps.vel_y);
  auto h_vel_z    = Kokkos::create_mirror_view(ps.vel_z);
  auto h_force_x  = Kokkos::create_mirror_view(ps.force_x);
  auto h_force_y  = Kokkos::create_mirror_view(ps.force_y);
  auto h_force_z  = Kokkos::create_mirror_view(ps.force_z);
  auto h_inv_mass = Kokkos::create_mirror_view(ps.inv_mass);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
  std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> mass_dist(0.5f, 2.0f);

  for (std::size_t i = 0; i < ps.size(); ++i) {
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

  for (std::size_t i = 0; i < std::min<std::size_t>(16, ps.size()); ++i) {
    h_inv_mass(i) = 0.0f;
  }

  Kokkos::deep_copy(ps.pos_x, h_pos_x);
  Kokkos::deep_copy(ps.pos_y, h_pos_y);
  Kokkos::deep_copy(ps.pos_z, h_pos_z);
  Kokkos::deep_copy(ps.vel_x, h_vel_x);
  Kokkos::deep_copy(ps.vel_y, h_vel_y);
  Kokkos::deep_copy(ps.vel_z, h_vel_z);
  Kokkos::deep_copy(ps.force_x, h_force_x);
  Kokkos::deep_copy(ps.force_y, h_force_y);
  Kokkos::deep_copy(ps.force_z, h_force_z);
  Kokkos::deep_copy(ps.inv_mass, h_inv_mass);
  Kokkos::fence();

  return timer.seconds() * 1000.0;
}

void print_sample_particle(const ParticleSystem& ps, int idx) {
  auto h_pos_x =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ps.pos_x);
  auto h_pos_y =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ps.pos_y);
  auto h_pos_z =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ps.pos_z);
  auto h_vel_x =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ps.vel_x);
  auto h_vel_y =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ps.vel_y);
  auto h_vel_z =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ps.vel_z);

  std::printf(
      "Sample particle %d: pos=(%.6f, %.6f, %.6f), vel=(%.6f, %.6f, %.6f)\n",
      idx, h_pos_x(idx), h_pos_y(idx), h_pos_z(idx), h_vel_x(idx), h_vel_y(idx),
      h_vel_z(idx));
}

template <class Policy>
void step_particles_with_policy(const char* label, const Policy& policy,
                                ParticleSystem& ps,
                                const SimulationParams& params) {
  auto pos_x    = ps.pos_x;
  auto pos_y    = ps.pos_y;
  auto pos_z    = ps.pos_z;
  auto vel_x    = ps.vel_x;
  auto vel_y    = ps.vel_y;
  auto vel_z    = ps.vel_z;
  auto force_x  = ps.force_x;
  auto force_y  = ps.force_y;
  auto force_z  = ps.force_z;
  auto inv_mass = ps.inv_mass;

  Kokkos::parallel_for(label, policy, KOKKOS_LAMBDA(const int i) {
    const float w = inv_mass(i);
    if (w == 0.0f) {
      return;
    }

    vel_x(i) += params.dt * force_x(i) * w;
    vel_y(i) += params.dt * force_y(i) * w;
    vel_z(i) += params.dt * force_z(i) * w;

    vel_x(i) *= params.damping;
    vel_y(i) *= params.damping;
    vel_z(i) *= params.damping;

    pos_x(i) += params.dt * vel_x(i);
    pos_y(i) += params.dt * vel_y(i);
    pos_z(i) += params.dt * vel_z(i);

    if (pos_y(i) < params.ground_y) {
      pos_y(i) = params.ground_y;
      if (vel_y(i) < 0.0f) {
        vel_y(i) = -vel_y(i) * params.restitution;
      }
      vel_x(i) *= params.ground_friction;
      vel_z(i) *= params.ground_friction;
    }
  });
}

void step_particles_baseline(ParticleSystem& ps, const SimulationParams& params) {
  Kokkos::RangePolicy<exec_space> policy(0, static_cast<int>(ps.size()));
  step_particles_with_policy("step_particles_baseline", policy, ps, params);
}

double benchmark_baseline(ParticleSystem& ps, int steps,
                          const SimulationParams& params) {
  Kokkos::fence();
  Kokkos::Timer timer;
  for (int s = 0; s < steps; ++s) {
    step_particles_baseline(ps, params);
  }
  Kokkos::fence();
  return timer.seconds() * 1000.0;
}

SimulationMetrics compute_metrics(const ParticleSystem& ps) {
  const auto pos_y    = ps.pos_y;
  const auto vel_x    = ps.vel_x;
  const auto vel_y    = ps.vel_y;
  const auto vel_z    = ps.vel_z;
  const auto inv_mass = ps.inv_mass;
  Kokkos::RangePolicy<exec_space> policy(0, static_cast<int>(ps.size()));

  float height_sum = 0.0f;
  Kokkos::parallel_reduce(
      "metrics_height_sum", policy,
      KOKKOS_LAMBDA(const int i, float& local_sum) { local_sum += pos_y(i); },
      height_sum);

  long long active_particles = 0;
  Kokkos::parallel_reduce(
      "metrics_active_particles", policy,
      KOKKOS_LAMBDA(const int i, long long& local_count) {
        if (inv_mass(i) != 0.0f) {
          ++local_count;
        }
      },
      active_particles);

  float total_kinetic_energy = 0.0f;
  Kokkos::parallel_reduce(
      "metrics_total_ke", policy,
      KOKKOS_LAMBDA(const int i, float& local_ke) {
        const float w = inv_mass(i);
        if (w != 0.0f) {
          const float mass = 1.0f / w;
          const float vx   = vel_x(i);
          const float vy   = vel_y(i);
          const float vz   = vel_z(i);
          const float v2   = vx * vx + vy * vy + vz * vz;
          local_ke += 0.5f * mass * v2;
        }
      },
      total_kinetic_energy);

  float max_speed_sq = 0.0f;
  Kokkos::parallel_reduce(
      "metrics_max_speed_sq", policy,
      KOKKOS_LAMBDA(const int i, float& local_max) {
        const float vx = vel_x(i);
        const float vy = vel_y(i);
        const float vz = vel_z(i);
        const float v2 = vx * vx + vy * vy + vz * vz;
        if (v2 > local_max) {
          local_max = v2;
        }
      },
      Kokkos::Max<float>(max_speed_sq));

  const float average_height = ps.size() > 0
                                   ? height_sum / static_cast<float>(ps.size())
                                   : 0.0f;
  return {average_height, total_kinetic_energy, std::sqrt(max_speed_sq),
          active_particles};
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr std::size_t particle_count = 1'000'000;
    constexpr int step_count             = 200;
    constexpr SimulationParams params{1.0f / 60.0f, 0.999f, 0.0f, 0.2f, 0.95f};
    ParticleSystem initial_state(particle_count);
    const double prepare_ms = initialize_particles(initial_state);

    ParticleSystem baseline_state(particle_count);
    Kokkos::deep_copy(baseline_state.pos_x, initial_state.pos_x);
    Kokkos::deep_copy(baseline_state.pos_y, initial_state.pos_y);
    Kokkos::deep_copy(baseline_state.pos_z, initial_state.pos_z);
    Kokkos::deep_copy(baseline_state.vel_x, initial_state.vel_x);
    Kokkos::deep_copy(baseline_state.vel_y, initial_state.vel_y);
    Kokkos::deep_copy(baseline_state.vel_z, initial_state.vel_z);
    Kokkos::deep_copy(baseline_state.force_x, initial_state.force_x);
    Kokkos::deep_copy(baseline_state.force_y, initial_state.force_y);
    Kokkos::deep_copy(baseline_state.force_z, initial_state.force_z);
    Kokkos::deep_copy(baseline_state.inv_mass, initial_state.inv_mass);

    const double baseline_ms =
        benchmark_baseline(baseline_state, step_count, params);
    const SimulationMetrics baseline_metrics = compute_metrics(baseline_state);

    std::printf("Particle count: %zu\n", particle_count);
    std::printf("Step count: %d\n", step_count);
    std::printf("Prepare data time: %.3f ms\n", prepare_ms);
    std::printf("Baseline total time: %.3f ms\n", baseline_ms);
    std::printf("Baseline per-step time: %.6f ms\n",
                baseline_ms / static_cast<double>(step_count));
    std::printf(
        "Baseline metrics: active=%lld avg_height=%.6f total_ke=%.6f "
        "max_speed=%.6f\n",
        baseline_metrics.active_particles, baseline_metrics.average_height,
        baseline_metrics.total_kinetic_energy, baseline_metrics.max_speed);
    print_sample_particle(initial_state, 123);
  }
  Kokkos::finalize();
  return 0;
}
