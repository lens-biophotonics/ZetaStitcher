major = 0
minor = 6
micro = 0

pre_release = None
post_release = None
dev_release = 4

__version__ = ''

if major is not None:
    __version__ += f'{major}'

if minor is not None:
    __version__ += f'.{minor}'

if micro is not None:
    __version__ += f'.{micro}'

if pre_release is not None:
    __version__ += f'{pre_release}'

if post_release is not None:
    __version__ += f'.post{post_release}'

if dev_release is not None:
    __version__ += f'.dev{dev_release}'
