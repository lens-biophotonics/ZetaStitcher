import semver

major = 0
minor = 1
patch = 1

prerelease = None
build = None

full_version = semver.format_version(major, minor, patch, prerelease, build)
