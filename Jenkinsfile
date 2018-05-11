#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    // parameters([booleanParam( name: 'push_image_to_docker_hub', defaultValue: false, description: 'Push rocsparse image to rocm docker-hub' )]),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// -- AUXILLARY HELPER FUNCTIONS
// import hudson.FilePath;
import java.nio.file.Path;

////////////////////////////////////////////////////////////////////////
// Construct the relative path of the build directory
void build_directory_rel( project_paths paths, compiler_data hcc_args )
{
  // if( hcc_args.build_config.equalsIgnoreCase( 'release' ) )
  // {
  //   paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name + '/release';
  // }
  // else
  // {
  //   paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name + '/debug';
  // }
  paths.project_build_prefix = paths.build_prefix + '/' + paths.project_name;

}

////////////////////////////////////////////////////////////////////////
// Lots of images are created above; no apparent way to delete images:tags with docker global variable
def docker_clean_images( String org, String image_name )
{
  // Check if any images exist first grepping for image names
  int docker_images = sh( script: "docker images | grep \"${org}/${image_name}\"", returnStatus: true )

  // The script returns a 0 for success (images were found )
  if( docker_images == 0 )
  {
    // run bash script to clean images:tags after successful pushing
    sh "docker images | grep \"${org}/${image_name}\" | awk '{print \$1 \":\" \$2}' | xargs docker rmi"
  }
}

////////////////////////////////////////////////////////////////////////
// -- BUILD RELATED FUNCTIONS

////////////////////////////////////////////////////////////////////////
// Checkout source code, source dependencies and update version number numbers
// Returns a relative path to the directory where the source exists in the workspace
void checkout_and_version( project_paths paths )
{
  paths.project_src_prefix = paths.src_prefix + '/' + paths.project_name

  dir( paths.project_src_prefix )
  {
    // checkout rocsparse
    checkout([
      $class: 'GitSCM',
      branches: scm.branches,
      doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
      extensions: scm.extensions + [[$class: 'CleanCheckout']],
      userRemoteConfigs: scm.userRemoteConfigs
    ])

    if( fileExists( 'CMakeLists.txt' ) )
    {
      def cmake_version_file = readFile( 'CMakeLists.txt' ).trim()
      //echo "cmake_version_file:\n${cmake_version_file}"

      cmake_version_file = cmake_version_file.replaceAll(/(\d+\.)(\d+\.)(\d+\.)\d+/, "\$1\$2\$3${env.BUILD_ID}")
      //echo "cmake_version_file:\n${cmake_version_file}"
      writeFile( file: 'CMakeLists.txt', text: cmake_version_file )
    }
  }

}

////////////////////////////////////////////////////////////////////////
// This creates the docker image that we use to build the project in
// The docker images contains all dependencies, including OS platform, to build
def docker_build_image( docker_data docker_args, project_paths paths )
{
  String build_image_name = "build"
  def build_image = null

  dir( paths.project_src_prefix )
  {
    def user_uid = sh( script: 'id -u', returnStdout: true ).trim()

    // Docker 17.05 introduced the ability to use ARG values in FROM statements
    // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
    // build_image = docker.build( "${paths.project_name}/${build_image_name}:latest", "--pull -f docker/${build_docker_file} --build-arg user_uid=${user_uid} --build-arg base_image=${from_image} ." )

    // JENKINS-44836 workaround by using a bash script instead of docker.build()
    sh "docker build -t ${paths.project_name}/${build_image_name}:latest -f docker/${docker_args.build_docker_file} ${docker_args.docker_build_args} --build-arg user_uid=${user_uid} --build-arg base_image=${docker_args.from_image} ."
    build_image = docker.image( "${paths.project_name}/${build_image_name}:latest" )
  }

  return build_image
}

// Docker related variables gathered together to reduce parameter bloat on function calls
class docker_data implements Serializable
{
  String from_image
  String build_docker_file
  String install_docker_file
  String docker_run_args
  String docker_build_args
}

// Docker related variables gathered together to reduce parameter bloat on function calls
class compiler_data implements Serializable
{
  String compiler_name
  String build_config
  String compiler_path
}

// Paths variables bundled together to reduce parameter bloat on function calls
class project_paths implements Serializable
{
  String project_name
  String src_prefix
  String project_src_prefix
  String build_prefix
  String project_build_prefix
  String build_command
}

// This defines a common build pipeline used by most targets
def build_pipeline( compiler_data compiler_args, docker_data docker_args, project_paths rocsparse_paths, def docker_inside_closure )
{
  ansiColor( 'vga' )
  {
    // NOTE: build_succeeded does not appear to be local to each function invokation.  I couldn't use it where each
    // node had a different success value.
    def build_succeeded = false;

    stage( "Build ${compiler_args.compiler_name} ${compiler_args.build_config}" )
    {
      // Checkout source code, dependencies and version files
      checkout_and_version( rocsparse_paths )

      // Conctruct a binary directory path based on build config
      build_directory_rel( rocsparse_paths, compiler_args );

      // Create/reuse a docker image that represents the rocsparse build environment
      def rocsparse_build_image = docker_build_image( docker_args, rocsparse_paths )

      // Print system information for the log
      rocsparse_build_image.inside( docker_args.docker_run_args, docker_inside_closure )

      // Build rocsparse inside of the build environment
      build_succeeded = docker_build_inside_image( rocsparse_build_image, compiler_args, docker_args, rocsparse_paths )
    }

    // After a successful build, test the installer
    // Only do this for rocm based builds
    if( compiler_args.compiler_name.toLowerCase( ).startsWith( 'hcc-' ) )
    {
      String job_name = env.JOB_NAME.toLowerCase( )
      String rocsparse_image_name = docker_test_install( compiler_args, docker_args, rocsparse_paths, job_name )

      docker_clean_images( job_name, rocsparse_image_name )
    }
  }
}

hcc_rocm:
{
  node( 'docker && rocm && dkms' )
  {
    def hcc_docker_args = new docker_data(
        from_image:'rocm/dev-ubuntu-16.04:1.7.1',
        build_docker_file:'dockerfile-build-ubuntu-16.04',
        install_docker_file:'dockerfile-rocsparse-ubuntu-16.04',
        docker_run_args:'--device=/dev/kfd --device=/dev/dri --group-add=video',
        docker_build_args:' --pull' )

    def hcc_compiler_args = new compiler_data(
        compiler_name:'hcc-rocm',
        build_config:'Release',
        compiler_path:'/opt/rocm/bin/hcc' )

    def rocsparse_paths = new project_paths(
        project_name:'rocsparse-hcc-rocm',
        src_prefix:'src',
        build_prefix:'src',
        build_command: './install.sh -cd' )

    def print_version_closure = {
      sh  """
          set -x
          /opt/rocm/bin/rocm_agent_enumerator -t ALL
          /opt/rocm/bin/hcc --version
        """
    }

    build_pipeline( hcc_compiler_args, hcc_docker_args, rocsparse_paths, print_version_closure )
  }
}
