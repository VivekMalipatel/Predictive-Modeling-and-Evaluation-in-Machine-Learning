import matplotlib.pyplot as plt
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

# Generate the coordinates for the ROC curve
outCurve = Utility.curve_coordinates (Y, 'Event', 'Non-Event', predProbEvent)

Threshold = outCurve['Threshold']
Precision = outCurve['Precision']
Recall = outCurve['Recall']

# Draw the Precision-Recall curve
plt.figure(dpi = 200)
plt.plot(Recall, Precision, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [6/11, 6/11], color = 'red', linestyle = ':', label = 'No Skill')
plt.grid(True)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

max_row = outCurve.loc[outCurve['F1Score'].idxmax()]
print( "the probability threshold that yields the highest F1 Score = ",max_row['Threshold'])