import logging
import logging.config


logging.config.fileConfig('logging.conf')
LOG = logging.getLogger('package_name')


def main():
    """Main function"""
    LOG.info("Hello world!")


if __name__ == '__main__':
    main()
