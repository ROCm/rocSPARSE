// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project)
{
    project.paths.construct_build_prefix()

    String compiler = platform.jenkinsLabel.contains('hip-clang') ? 'hipcc' : 'hcc'
    String hipClangArgs = platform.jenkinsLabel.contains('hip-clang') ? ' --hip-clang' : ''
    String sudo = platform.jenkinsLabel.contains('sles') ? 'sudo' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib CXX=/opt/rocm/bin/${compiler} ${project.paths.build_command} ${hipClangArgs}
            """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release/clients/staging
                ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
            """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command

    if(platform.jenkinsLabel.contains('centos'))
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                ${sudo} make package
                ${sudo} mkdir -p package
                ${sudo} mv *.rpm package/
                ${sudo} rpm -qlp package/*.rpm
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")
    }
    else if(platform.jenkinsLabel.contains('hip-clang') || platform.jenkinsLabel.contains('sles'))
    {
        packageCommand = null
    }
    else
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                make package_clients
                mkdir -p package
                mv *.deb package/
                mv clients/*.deb package/
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
    }
}

return this
