pipeline {
    agent none

    options {
        disableConcurrentBuilds(abortPrevious: true)
        timeout(time: 8, unit: 'HOURS')
        timestamps()
    }

    parameters {
        string(name: 'KOKKOS_SDK_PLATFORM', defaultValue: 'linux-x86_64-gcc13', description: 'Platform triplet recorded in the SDK name and manifest')
        string(name: 'KOKKOS_SDK_CUDA_ARCHES', defaultValue: 'VOLTA70,AMPERE80,HOPPER90,BLACKWELL100', description: 'Comma-separated Kokkos CUDA architecture tokens for the single CUDA variant')
        booleanParam(name: 'BUILD_CUDA', defaultValue: true, description: 'Build the CUDA Kokkos SDK variant')
        booleanParam(name: 'BUILD_MACA', defaultValue: true, description: 'Build the MACA Kokkos SDK variant')
        booleanParam(name: 'PUBLISH_GITEA_PACKAGE', defaultValue: false, description: 'Publish the assembled SDK tarball to a Gitea Generic Package Registry')
        string(name: 'GITEA_BASE_URL', defaultValue: 'https://gitea.example.com', description: 'Gitea server base URL')
        string(name: 'GITEA_PACKAGE_OWNER', defaultValue: 'wjw03', description: 'Gitea package owner or organization')
        string(name: 'GITEA_TOKEN_CREDENTIALS_ID', defaultValue: 'gitea-package-token', description: 'Jenkins secret text credential containing a Gitea package token')
    }

    environment {
        SDK_WORK_DIR = 'out/kokkos-sdk'
        SDK_TARBALL_DIR = 'out/kokkos-sdk-artifacts'
    }

    stages {
        stage('Build Variants') {
            parallel {
                stage('cpu') {
                    agent {
                        dockerfile {
                            filename 'Dockerfile.gcc'
                            dir 'scripts/docker'
                            label 'docker'
                            args '--env NODE_NAME=${env.NODE_NAME} --env STAGE_NAME=${env.STAGE_NAME}'
                        }
                    }
                    steps {
                        checkout scm
                        sh '''#!/usr/bin/env bash
                              set -euo pipefail
                              scripts/package-kokkos-sdk.sh build \
                                --backend cpu \
                                --platform "${KOKKOS_SDK_PLATFORM}" \
                                --work-dir "${SDK_WORK_DIR}" \
                                --shared'''
                        stash name: 'kokkos-sdk-stage-cpu', includes: 'out/kokkos-sdk/stage/cpu/**'
                    }
                }

                stage('cuda') {
                    when {
                        expression { return params.BUILD_CUDA }
                    }
                    agent {
                        dockerfile {
                            filename 'Dockerfile.nvcc'
                            dir 'scripts/docker'
                            additionalBuildArgs '--build-arg BASE=nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04@sha256:5f603101462baa721ff6ddc44af82f6e9ba7cbd92a424c9f9f348e6e9d6d64c3 --build-arg ADDITIONAL_PACKAGES="gfortran clang" --build-arg CMAKE_VERSION=3.25.3'
                            label 'nvidia-docker'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES --env NODE_NAME=${env.NODE_NAME} --env STAGE_NAME=${env.STAGE_NAME}'
                        }
                    }
                    environment {
                        NVCC_WRAPPER_DEFAULT_COMPILER = 'g++-11'
                    }
                    steps {
                        checkout scm
                        sh '''#!/usr/bin/env bash
                              set -euo pipefail
                              scripts/package-kokkos-sdk.sh build \
                                --backend cuda \
                                --platform "${KOKKOS_SDK_PLATFORM}" \
                                --cuda-arch-list "${KOKKOS_SDK_CUDA_ARCHES}" \
                                --work-dir "${SDK_WORK_DIR}" \
                                --shared'''
                        stash name: 'kokkos-sdk-stage-cuda', includes: 'out/kokkos-sdk/stage/cuda/**'
                    }
                }

                stage('maca') {
                    when {
                        expression { return params.BUILD_MACA }
                    }
                    agent {
                        label 'maca'
                    }
                    steps {
                        checkout scm
                        sh '''#!/usr/bin/env bash
                              set -euo pipefail
                              scripts/package-kokkos-sdk.sh build \
                                --backend maca \
                                --platform "${KOKKOS_SDK_PLATFORM}" \
                                --work-dir "${SDK_WORK_DIR}" \
                                --shared'''
                        stash name: 'kokkos-sdk-stage-maca', includes: 'out/kokkos-sdk/stage/maca/**'
                    }
                }
            }
        }

        stage('Assemble SDK') {
            agent {
                dockerfile {
                    filename 'Dockerfile.gcc'
                    dir 'scripts/docker'
                    label 'docker'
                    args '--env NODE_NAME=${env.NODE_NAME} --env STAGE_NAME=${env.STAGE_NAME}'
                }
            }
            steps {
                checkout scm
                unstash 'kokkos-sdk-stage-cpu'
                script {
                    if (params.BUILD_CUDA) {
                        unstash 'kokkos-sdk-stage-cuda'
                    }
                    if (params.BUILD_MACA) {
                        unstash 'kokkos-sdk-stage-maca'
                    }
                    def variants = ['cpu']
                    if (params.BUILD_CUDA) {
                        variants.add('cuda')
                    }
                    if (params.BUILD_MACA) {
                        variants.add('maca')
                    }
                    env.KOKKOS_SDK_BACKENDS_FOR_ASSEMBLY = variants.join(',')
                }
                sh '''#!/usr/bin/env bash
                      set -euo pipefail
                      scripts/package-kokkos-sdk.sh assemble \
                        --backend "${KOKKOS_SDK_BACKENDS_FOR_ASSEMBLY}" \
                        --platform "${KOKKOS_SDK_PLATFORM}" \
                        --work-dir "${SDK_WORK_DIR}"

                      mkdir -p "${SDK_TARBALL_DIR}"
                      sdk_dir="$(find "${SDK_WORK_DIR}" -maxdepth 1 -type d -name 'kokkos-sdk-*' | sort | tail -1)"
                      sdk_name="$(basename "${sdk_dir}")"
                      tar -C "$(dirname "${sdk_dir}")" -czf "${SDK_TARBALL_DIR}/${sdk_name}.tar.gz" "${sdk_name}"
                      sha256sum "${SDK_TARBALL_DIR}/${sdk_name}.tar.gz" > "${SDK_TARBALL_DIR}/${sdk_name}.tar.gz.sha256"
                      printf '%s\n' "${sdk_name}" > "${SDK_TARBALL_DIR}/sdk-name.txt"'''
                archiveArtifacts artifacts: 'out/kokkos-sdk-artifacts/*', fingerprint: true
            }
        }

        stage('Publish Gitea Package') {
            when {
                expression { return params.PUBLISH_GITEA_PACKAGE }
            }
            agent {
                docker {
                    image 'curlimages/curl:8.10.1'
                    label 'docker'
                }
            }
            steps {
                checkout scm
                unarchive mapping: ['out/kokkos-sdk-artifacts/*': 'out/kokkos-sdk-artifacts']
                withCredentials([string(credentialsId: params.GITEA_TOKEN_CREDENTIALS_ID, variable: 'GITEA_TOKEN')]) {
                    sh '''#!/usr/bin/env sh
                          set -eu
                          sdk_name="$(cat out/kokkos-sdk-artifacts/sdk-name.txt)"
                          sdk_version="${sdk_name#kokkos-sdk-}"
                          sdk_version="${sdk_version%-${KOKKOS_SDK_PLATFORM}}"
                          package_url="${GITEA_BASE_URL}/api/packages/${GITEA_PACKAGE_OWNER}/generic/kokkos-sdk/${sdk_version}/${sdk_name}.tar.gz"
                          checksum_url="${GITEA_BASE_URL}/api/packages/${GITEA_PACKAGE_OWNER}/generic/kokkos-sdk/${sdk_version}/${sdk_name}.tar.gz.sha256"
                          curl --fail --show-error --location --user "token:${GITEA_TOKEN}" --upload-file "out/kokkos-sdk-artifacts/${sdk_name}.tar.gz" "${package_url}"
                          curl --fail --show-error --location --user "token:${GITEA_TOKEN}" --upload-file "out/kokkos-sdk-artifacts/${sdk_name}.tar.gz.sha256" "${checksum_url}"'''
                }
            }
        }
    }
}
