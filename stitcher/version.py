major = 0
minor = 1
patch = 0

full_version = '{}.{}'.format(major, minor)
if patch:
    full_version = '{}.{}'.format(full_version, patch)
