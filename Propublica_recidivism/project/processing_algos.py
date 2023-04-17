from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.preprocessing import Reweighing, OptimPreproc
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools


class Processing:

  def __init__(self, privileged_groups, unprivileged_groups): 
    self.privileged_groups = privileged_groups
    self.unprivileged_groups = unprivileged_groups


  #---------------------------------------------------- Pre-Processing algorithm
  def reweighing_processing(self, train_data):
    processor = Reweighing(self.unprivileged_groups, self.privileged_groups)
    processor.fit(train_data)
    return processor.transform(train_data)
  

  def optim_pre_processing(self, train_data, optim_options):
    processor = OptimPreproc(OptTools, optim_options, privileged_groups = self.privileged_groups,
                            unprivileged_groups = self.unprivileged_groups)
    processor.fit(train_data)
    transformed_data = processor.transform(train_data, transform_Y=True)
    transformed_data = train_data.align_datasets(transformed_data)
    return transformed_data


  #----------------------------------------------------- In-Processing algorithm
  def adversarial_debiasing_processing(self, train_data, test_data, session, scope_name: str = 'plain_classifier', debias: bool = False):
    adv_model = AdversarialDebiasing(privileged_groups = self.privileged_groups,
                          unprivileged_groups = self.unprivileged_groups,
                          scope_name = scope_name,
                          debias = debias,
                          sess = session)

    adv_model.fit(train_data)
    return adv_model.predict(train_data), adv_model.predict(test_data) 


  #--------------------------------------------------- Post-Processing algorithm

  def calibrated_eq_odds_processing(self, train_data, train_predictions, test_data, cost_constraint, seed):
    processor = CalibratedEqOddsPostprocessing(privileged_groups = self.privileged_groups,
                                     unprivileged_groups = self.unprivileged_groups,
                                     cost_constraint = cost_constraint,
                                     seed = seed)
    processor = processor.fit(train_data, train_predictions)
    transformed_train_pred = processor.predict(train_data)
    transformed_test_pred = processor.predict(test_data)
    return transformed_train_pred, transformed_test_pred


    
