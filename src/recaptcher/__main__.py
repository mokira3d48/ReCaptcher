import os
import logging
import logging.config

from .dataset import DatasetCollector, DatasetVocabBuilder
from .training import Main as Training
from .predictions import Prediction
from .utils import load_args


logging.config.fileConfig('logging.conf')
LOG = logging.getLogger('alt')


actions = {
    'DatasetCollector': DatasetCollector,
    'DatasetVocabBuilder': DatasetVocabBuilder,
    'Training': Training,
    'Prediction': Prediction,
}


def main():
    """Main function"""
    if len(os.sys.argv) <= 1:
        print("No argument file given!")
        return

    args_file = os.sys.argv[1]
    args = load_args(args_file)

    action_class = actions.get(args['action'])
    if not action_class:
        LOG.error("No action named " + str(args['action']) + " found.")
        return

    action = action_class(**args)
    action.run()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Bye!")
        os.sys.exit(125)
