#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@clang9') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 22 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])


////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rocSPARSECI:
{

    def rocsparse = new rocProject('rocsparse')
    // customize for project
    rocsparse.paths.build_command = './install.sh -c'
    rocsparse.compiler.compiler_name = 'c++'
    rocsparse.compiler.compiler_path = 'c++'
    
    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx900 && centos7', 'gfx906'], rocsparse)

    boolean formatCheck = true

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        
        def command
        
        if(platform.jenkinsLabel.contains('centos'))
        {   
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    export PATH=/opt/rocm/hsa/include:$PATH
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rh/devtoolset-7/root/usr/bin/c++ ${project.paths.build_command}
                """
        }
        else
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=${project.compiler.compiler_path} ${project.paths.build_command}
                """
        }
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def command
        
        if(platform.jenkinsLabel.contains('centos'))
        {
            if(auxiliary.isJobStartedByTimer())
            {
                command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release/clients/tests
                        LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG sudo ./rocsparse-test --gtest_output=xml --gtest_color=yes #--gtest_filter=*nightly*-*known_bug* #--gtest_filter=*nightly*
                    """
            }
            else
            {
                command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release/clients/tests
                        LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG sudo ./rocsparse-test --gtest_output=xml --gtest_color=yes #--gtest_filter=*quick*:*pre_checkin*-*known_bug* #--gtest_filter=*checkin*
                    """
            }
        }           
        else
        {
            if(auxiliary.isJobStartedByTimer())
            {
                command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release/clients/tests
                        LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml --gtest_color=yes #--gtest_filter=*nightly*-*known_bug* #--gtest_filter=*nightly*
                    """
            }
            else
            {
                command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release/clients/tests
                        LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml --gtest_color=yes #--gtest_filter=*quick*:*pre_checkin*-*known_bug* #--gtest_filter=*checkin*
                    """
            }
        }

        platform.runCommand(this, command)
        junit "${project.paths.project_build_prefix}/build/release/clients/tests/*.xml"
    }

    def packageCommand =
    {
        platform, project->

        def command 
        
        if(platform.jenkinsLabel.contains('centos'))
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make package
                    rm -rf package && mkdir -p package
                    mv *.rpm package/
                    rpm -qlp package/*.rpm
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")        
        }
        else
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make package
                    rm -rf package && mkdir -p package
                    mv *.deb package/
                    dpkg -c package/*.deb
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
        }
    }

    buildProject(rocsparse, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
