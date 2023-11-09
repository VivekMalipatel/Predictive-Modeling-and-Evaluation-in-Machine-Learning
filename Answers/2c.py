import numpy
import Utility

import warnings
warnings.filterwarnings("ignore")

Y = numpy.array(['Non-Event',
                 'Non-Event',
                 'Non-Event',
                 'Non-Event',
                 'Non-Event',
                 'Non-Event',
                 'Non-Event',
                 'Non-Event',
                 'Non-Event',
                 'Non-Event',
                 'Event',
                 'Event',
                 'Event',
                 'Event',
                 'Event',
                 'Event',
                 'Event',
                 'Event',
                 'Event',
                 'Event'])

predProbEvent = numpy.array([0.0814,0.1197,0.1969,0.3505,0.3878,0.3940,0.4828,0.4889,0.5587,0.6175,0.4974,0.6732,0.6744,0.6836,0.7475,0.7828,0.6342,0.6527,0.6668,0.5614])
p = numpy.array([0.6342,0.4974])

outSeriesa = Utility.binary_model_metric (Y, 'Event', 'Non-Event', predProbEvent, p[0])
print('Misclassification Rate for parts (a) : {:.13f}' .format(outSeriesa['MCE']))

outSeriesb = Utility.binary_model_metric (Y, 'Event', 'Non-Event', predProbEvent, p[1])
print('Misclassification Rate for parts (b) : {:.13f}' .format(outSeriesb['MCE']))