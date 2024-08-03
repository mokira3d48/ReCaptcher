import re
from pgit import ProgressIter


def get_variables(string):
	"""Funtion to get variable from string formatting

	:param string: The initial string.
	:type string: str
	:rtype: `list` of `str`
	"""
	pattern = r"{([^}]*)}"
	return re.findall(pattern, string)


class Progress(ProgressIter):

    def __init__(self, length, bins, log_format = '', **kwargs):
        super().__init__(length=length, bins=bins, **kwargs)
        self._loginfo = {}
        self._log_format = log_format
        self._init_data()

    @property
    def loginfo(self):
        """:str: the description string format"""
        return self._loginfo

    @loginfo.setter
    def loginfo(self, value):
        self._loginfo = value

    def _init_data(self):
        """Data initialization method"""
        variables = get_variables(self._log_format)
        for var in variables:
            vartypes = var.split(':')
            varname = ''
            vartype = ''
            if len(vartypes) == 2:
                varname = vartypes[0]
                vartype = vartypes[1]
            elif len(vartypes) == 1:
                varname = vartypes[0]
            else:
                continue

            if not vartype or 's' in vartype:
                self._loginfo[varname] = ''
            else:
                self._loginfo[varname] = 0

    def log(self, attrname = None, message = None):
        if attrname and message:
            self._loginfo[attrname] = message

        super().log(self._log_format.format(**self._loginfo))


def main():
    """Main function"""
    import time
    pbar_format = "{logger:32s} {pbar} [\033[91m{purcent:6.2f}\033[0m - {time_rem}]"
    pbar = Progress(
        1000,
        bins=20,
        log_format="{train_loss:.2f}\t{valid_loss}\t{epoch}/{epochs}",
        barf=pbar_format,
    )
    # pbar.loginfo = {'train_loss': 0.0, 'valid_loss': 0.0, 'epoch': 0, 'epochs': 1000}
    print(pbar.loginfo)
    for i in range(1000):
        pbar.loginfo['epoch'] = i+1
        pbar.loginfo['train_loss'] = float(i) + 0.45
        pbar.log()
        pbar.step(1)
        time.sleep(0.1)


if __name__ == '__main__':
    main()
