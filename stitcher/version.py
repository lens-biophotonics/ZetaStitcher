import semver

major = 0
minor = 2
patch = 0

prerelease = None
build = None

full_version = semver.format_version(major, minor, patch, prerelease, build)
