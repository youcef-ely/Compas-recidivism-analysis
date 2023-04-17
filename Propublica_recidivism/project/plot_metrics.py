import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np






def confusion_matrix(confusion_matrix_before, confusion_matrix_after, model_name):

  xticklabels = ['Negative', 'Positive']
  yticklabels = ['Negative', 'Positive']
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
  plt.subplots_adjust(top = 0.85)

  names = ['TN', 'FP', 'FN', 'TP']
  real_numbers = [number for number in confusion_matrix_before.reshape(confusion_matrix_before.size, 1)]
  ratio = [np.round(number / confusion_matrix_before.sum(), 2) for number in confusion_matrix_before.reshape(confusion_matrix_before.size, 1)]
  labels = [name + '\n' + str(num[0]) + '\n' + str(round(ratio[0] * 100, 4)) + "%" for name, num, ratio in zip(names, real_numbers, ratio)]
  labels = np.asarray(labels).reshape(2, 2)
  sns.heatmap(confusion_matrix_before, annot = np.asarray(labels).reshape(2, 2), 
              xticklabels = xticklabels, yticklabels = yticklabels, ax = ax1, fmt = '', cmap = 'crest', linewidths = .5)
  ax1.set_title('Before mitigation')


  names = ['TN', 'FP', 'FN', 'TP']
  real_numbers = [number for number in confusion_matrix_after.reshape(confusion_matrix_after.size, 1)]
  ratio = [np.round(number / confusion_matrix_after.sum(), 2) for number in confusion_matrix_after.reshape(confusion_matrix_after.size, 1)]
  labels = [name + '\n' + str(num[0]) + '\n' + str(round(ratio[0] * 100, 4)) + "%" for name, num, ratio in zip(names, real_numbers, ratio)]
  labels = np.array(labels).reshape(2, 2)
  sns.heatmap(confusion_matrix_after, annot = labels, xticklabels = xticklabels, 
              yticklabels = yticklabels, ax = ax2, fmt = '', cmap = 'crest', linewidths = .5)

  ax2.set_title('After mitigation')

  fig.suptitle('Confusion Matrix for {}'.format(model_name), fontsize = 25)
  #fig.savefig('CM ' + model_name + '.png')
  
  plt.legend()



def plot_fairness_metrics(mean_diff, smoothed_empirical, disparate_impact, comp, processing_techinque):
  
  labels = ['Original']
  if comp: 
    labels.append('Mitigated') 

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 5))
  plt.subplots_adjust(top=0.85)

  #---------------------------------------------------------------------------
  ax1.set_title('Statistical Parity Difference', fontsize=10)
  ax1.bar(labels, mean_diff, color =['black', 'cyan'], width = 0.5)
  ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
  ax1.axhline(0, color = 'grey', linewidth = 0.8)
  ax1.axhline(y = 0, color = 'g', linestyle = 'dashed', label = "Fair")
  ax1.legend(bbox_to_anchor = (1, 1), loc = 'upper right')

  for i in range(len(labels)):
      offset = 0.1
      if mean_diff[i] < 0 : 
        offset = -offset
      ax1.text(i, mean_diff[i] + offset, "{:.4f}".format(mean_diff[i]), ha = 'center', fontsize = 8)

  ax2.set_title('Equal Opportunity Difference', fontsize=10)
  ax2.bar(labels, smoothed_empirical, color =['black', 'cyan'], width = 0.5)
  ax2.set_yticks([-1,-0.5,0,0.5,1])
  ax2.axhline(0, color='grey', linewidth=0.8)
  ax2.axhline(y = 0, color = 'g', linestyle = 'dashed', label = "Fair")
  ax2.legend(bbox_to_anchor = (1, 1), loc = 'upper right')

  for i in range(len(labels)):
      offset = 0.1
      if smoothed_empirical[i] < 0 : offset =  -offset
      ax2.text(i, smoothed_empirical[i] + offset, "{:.4f}".format(smoothed_empirical[i]), ha = 'center', fontsize = 8)

  ax3.set_title('Disparate Impact', fontsize=10)
  ax3.bar(labels, disparate_impact, color = ['black', 'cyan'], width = 0.5)
  ax3.set_yticks([0, 0.5, 1, 1.5])
  ax3.axhline(0, color = 'grey', linewidth=0.8)
  ax3.axhline(y = 1, color = 'g', linestyle = 'dashed', label = "Fair")
  ax3.legend(bbox_to_anchor = (1, 1), loc = 'upper right')

  for i in range(len(labels)):
      offset = 0.05
      if disparate_impact[i] < 0 : offset =  -offset
      ax3.text(i, disparate_impact[i] + offset, "{:.4f}".format(disparate_impact[i]), ha = 'center', fontsize = 8)
  
  fig.suptitle('Fairness Metrics {} '.format(processing_techinque), fontsize = 25)
  #fig.savefig('/content/' + processing_techinque + '.png')
  plt.show()


def plot_protected_attribute(data, protected_attribute):
  n = len(data[protected_attribute])
  ratios = data[protected_attribute].value_counts().sort_values()
  ratios_n = np.round(data[protected_attribute].value_counts(normalize = True).sort_values(), 3)
  pal = sns.color_palette("crest" ,len(ratios))
  fig, ax1 = plt.subplots(figsize = (12, 10))
  sns.barplot(list(ratios.index), list(ratios.values), palette = pal, ax = ax1)
  for i in range(len(ratios)):
      offset = 10
      ax1.text(i, ratios[i] + 40, "{} ({}%)".format(ratios[i],round(ratios_n[i]*100, 2)), ha = 'center', fontsize = 15)
  plt.title(protected_attribute.capitalize() + ' Partition', fontsize = 25)
  #fig.savefig('/content/bars.png')