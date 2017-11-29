major = 0
minor = 3
micro = 2

pre_release = ''
post_release = ''
dev_release = ''

__version__ = ''

if major != '':
    __version__ += '{}'.format(major)

if minor != '':
    __version__ += '.{}'.format(minor)

if micro != '':
    __version__ += '.{}'.format(micro)

if pre_release != '':
    __version__ += '{}'.format(pre_release)

if post_release != '':
    __version__ += '.post{}'.format(post_release)

if dev_release != '':
    __version__ += '.dev{}'.format(dev_release)
