# -*- coding: utf-8 -*.t-
import os
from setuptools import find_packages, setup, Extension

from Cython.Distutils import build_ext as orig_build_ext
from Cython.Build import cythonize

from distutils.command.build_clib import build_clib
from distutils.dep_util import newer_group
import distutils.dist
from distutils import log
import distutils.command.build


distutils.dist.Distribution.cuda_sources = None
distutils.dist.Distribution.cuda_gpu_arch = None


def has_nvcc(self):
    return self.distribution.cuda_sources


distutils.command.build.build.sub_commands.insert(0, ('build_nvcc', has_nvcc))
orig_build_ext.sub_commands.insert(0, ('build_nvcc', has_nvcc))


class build_nvcc(build_clib):

    def run(self):
        return build_clib.run(self)

    def finalize_options(self):
        # almost verbatim copy of the build_clib.finalize_options()
        self.set_undefined_options('build',
                                   ('build_temp', 'build_clib'),
                                   ('build_temp', 'build_temp'),
                                   ('compiler', 'compiler'),
                                   ('debug', 'debug'),
                                   ('force', 'force'))

        # commended out from build_clib.finalize_options()
#        self.libraries = self.distribution.libraries
#        if self.libraries:
#            self.check_library_list(self.libraries)

        if self.include_dirs is None:
            self.include_dirs = self.distribution.include_dirs or []
        if isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)

        self.libraries = self.distribution.cuda_sources

    def build_libraries(self, libraries):
        self.compiler.src_extensions.append('.cu')

        arch = self.distribution.cuda_gpu_arch
        arch_arg = ["-arch=%s" % arch] if arch else []

#        args = ["nvcc", "--device-c", "--ptxas-options=-v", "-c",
        args = ["nvcc", "--device-c", "-c",
                "--compiler-options", ','.join(self.compiler.compiler_so[1:])] + arch_arg

        self.compiler.set_executables(compiler_so=args)

#        args = ["nvcc", "--device-c", "--ptxas-options=-v", "-c", "-Xptxas",  "-v"
        args = ["nvcc", "--device-c", "-c", "-Xptxas", "-v",
                "--compiler-options", ','.join(self.compiler.compiler_so[1:])] + arch_arg

        self.compiler.set_executables(compiler_cxx=args)

        args = ([
            'nvcc', "--lib",
            "--compiler-options", ','.join(self.compiler.archiver[1:])] + arch_arg +
            ["--output-file"])  # --output-file should be last one

        self.compiler.set_executables(archiver=args)
        sources = self.get_source_files()
        deps = sum([lib[1].get('depends', []) for lib in self.libraries], [])
        if not self.force:
            for name in self.get_library_names():
                target = self.compiler.library_filename(
                    name, output_dir=self.build_clib)

                if newer_group(sources + deps, target, 'newer'):
                    break
            else:
                log.debug("skipping nvcc compiler (up-to-date)")
                return

        return build_clib.build_libraries(self, libraries)


class build_ext(orig_build_ext):
    def run(self):
        if has_nvcc(self):
            # run clib build. Or, clib won't be build with such command as 'pip
            # install -e .'

            build_clib = self.get_finalized_command('build_nvcc')
            build_clib.force = self.force
            build_clib.compiler = None  # bug in distutils?

            build_clib.run()

            clibdir = build_clib.build_clib
            self.libraries.extend(build_clib.get_library_names() or [])
            self.library_dirs.append(clibdir)

            deps = [
                build_clib.compiler.library_filename(name, output_dir=clibdir)
                for name in build_clib.get_library_names()]

            for ext in self.extensions:
                ext.depends += deps

        orig_build_ext.run(self)

    def build_extensions(self):
        arch = self.distribution.cuda_gpu_arch
        arch_arg = ["-arch=%s" % arch] if arch else []
        self.compiler.set_executables(linker_so=[
            'nvcc', "--compiler-options", "-fPIC,-shared,-pthread"] + arch_arg)

        # distutils uses compiler_cxx[0] for linker if language is c++.
        self.compiler.set_executables(compiler_cxx=[
            'nvcc', "--compiler-options", "-fPIC,-shared,-pthread"] + arch_arg)

        orig_build_ext.build_extensions(self)


requires = [
    "numpy", "scikit-image", "scikit-learn", "Cython>=0.24.0", "Pillow", "future"
]


IN_CI = False
if os.environ.get('CI', None):
    IN_CI = True   # In gitlab CI


ext_modules = []
cuda_sources = []


cuda_depends = ['renom/cuda/thrust/thrust_func_defs.pxi',
                'renom/cuda/thrust/thrust_funcs.pxi',
                'renom/cuda/thrust/thrust_funcs.h'
                ]


def setup_cuda():
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not cuda_home or not os.path.exists(cuda_home):
        # CUDA is not installed
        print('Failed to detect cuda.')
        return

    libraries = [
        os.path.join(cuda_home, "lib64")
    ]

    includes = [
        os.path.join(cuda_home, "include"),
    ]

    ext_base = Extension('renom.cuda.base.cuda_base',
                         sources=['renom/cuda/base/cuda_base.pyx'],
                         depends=cuda_depends,
                         libraries=['cublas', 'cuda', 'cudart', 'nvToolsExt'],
                         library_dirs=libraries,
                         language='c++',
                         include_dirs=includes,
                         )

    ext_utils = Extension('renom.cuda.base.cuda_utils',
                          sources=['renom/cuda/base/cuda_utils.pyx'],
                          depends=cuda_depends,
                          libraries=['cublas', 'cuda', 'cudart'],
                          library_dirs=libraries,
                          language='c++',
                          include_dirs=includes,
                          )

    ext_cublas = Extension('renom.cuda.cublas.cublas',
                           sources=['renom/cuda/cublas/cublas.pyx'],
                           depends=cuda_depends,
                           libraries=['cublas', 'cuda', 'cudart'],
                           library_dirs=libraries,
                           language='c++',
                           include_dirs=includes,
                           )

    ext_cudnn = Extension('renom.cuda.cudnn.cudnn',
                          sources=['renom/cuda/cudnn/cudnn.pyx'],
                          depends=cuda_depends,
                          libraries=['cublas', 'cuda', 'cudart', 'cudnn'],
                          library_dirs=libraries,
                          language='c++',
                          include_dirs=includes,
                          )

    ext_curand = Extension('renom.cuda.curand.curand',
                           sources=['renom/cuda/curand/curand.pyx'],
                           depends=cuda_depends,
                           libraries=['curand', 'cuda', 'cudart'],
                           library_dirs=libraries,
                           language='c++',
                           include_dirs=includes,
                           )

    ext_thrust_float = Extension('renom.cuda.thrust.thrust_float',
                                 sources=['renom/cuda/thrust/thrust_float.pyx'],
                                 depends=cuda_depends,
                                 libraries=['cublas', 'cuda', 'cudart'],
                                 library_dirs=libraries,
                                 language='c++',
                                 include_dirs=includes,
                                 )

    ext_thrust_double = Extension('renom.cuda.thrust.thrust_double',
                                  sources=['renom/cuda/thrust/thrust_double.pyx'],
                                  depends=cuda_depends,
                                  libraries=['cublas', 'cuda', 'cudart'],
                                  library_dirs=libraries,
                                  language='c++',
                                  include_dirs=includes,
                                  )

    ext_gpuvalue = Extension('renom.cuda.gpuvalue.gpuvalue',
                             sources=['renom/cuda/gpuvalue/gpuvalue.py'],
                             depends=cuda_depends,
                             libraries=['cublas', 'cuda', 'cudart'],
                             library_dirs=libraries,
                             language='c++',
                             include_dirs=includes,
                             )

    global ext_modules, cuda_sources

    ext_modules = [ext_base,
                   ext_utils,
                   ext_cublas,
                   ext_cudnn,
                   ext_curand,
                   ext_thrust_float,
                   ext_thrust_double,
                   ext_gpuvalue,
                   ]

    cuda_sources = [('cuda_misc_a',
                     {'sources': [
                         'renom/cuda/thrust/thrust_funcs_float.cu',
                         'renom/cuda/thrust/thrust_funcs_double.cu', ],
                      'depends': [
                         'renom/cuda/thrust/thrust_funcs_double.h',
                         'renom/cuda/thrust/thrust_funcs_float.h',
                         'renom/cuda/thrust/thrust_funcs.inl'],
                      })]


setup_cuda()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

import numpy
setup(
    install_requires=requires,
    cuda_sources=cuda_sources,
    cuda_gpu_arch='sm_30',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext, 'build_nvcc': build_nvcc},
    name='renom',
    packages=find_packages(),
    include_dirs=[numpy.get_include()],
    version='2.7.3')
