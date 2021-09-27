library(tidymodels)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(pROC)
library(vip)

# import ------------------------------------------------------------------


# Bases de dados ----------------------------------------------------------

ga_dataset <- readRDS("ga_dataset.rds")
glimpse(ga_dataset) # German Risk

ga_dataset %>% count(target)

# Base de treino e teste --------------------------------------------------

set.seed(1)
ga_initial_split <- initial_split(ga_dataset, strata = "target", prop = 0.75)

ga_train <- training(ga_initial_split)
ga_test  <- testing(ga_initial_split)

# Reamostragem ------------------------------------------------------------

ga_resamples <- vfold_cv(ga_train, v = 5, strata = "target")

# Exploratória ------------------------------------------------------------

# skimr::skim(ga_train)
# visdat::vis_miss(ga_train)
# ga_train %>%
#   select(where(is.numeric)) %>%
#   cor(use = "pairwise.complete.obs") %>%
#   corrplot::corrplot()

# Árvore de decisão -------------------------------------------------------

## Data prep

ga_dt_recipe <- recipe(target ~ ., data = ga_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_zv(all_predictors())

## Modelo

ga_dt_model <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("rpart")

## Workflow

ga_dt_wf <- workflow() %>%
  add_model(ga_dt_model) %>%
  add_recipe(ga_dt_recipe)

## Tune

grid_dt <- grid_random(
  cost_complexity(c(-9, -2)),
  tree_depth(range = c(5, 15)),
  min_n(range = c(20, 40)),
  size = 20
)

# doParallel::registerDoParallel(4)

ga_dt_tune_grid <- tune_grid(
  ga_dt_wf,
  resamples = ga_resamples,
  grid = grid_dt,
  metrics = metric_set(roc_auc)
)

# doParallel::stopImplicitCluster()

autoplot(ga_dt_tune_grid)
collect_metrics(ga_dt_tune_grid)


# Desempenho dos modelos finais ----------------------------------------------

ga_dt_best_params <- select_best(ga_dt_tune_grid, "roc_auc")
ga_dt_wf <- ga_dt_wf %>% finalize_workflow(ga_dt_best_params)
ga_dt_last_fit <- last_fit(ga_dt_wf, ga_initial_split)


ga_test_preds <- collect_predictions(ga_dt_last_fit) %>% mutate(modelo = "dt")

## roc
ga_test_preds %>%
  group_by(modelo) %>%
  roc_curve(target, .pred_bad) %>%
  autoplot()

## lift
ga_test_preds %>%
  group_by(modelo) %>%
  lift_curve(target, .pred_bad) %>%
  autoplot()

# Variáveis importantes
ga_dt_last_fit_model <- ga_dt_last_fit$.workflow[[1]]$fit$fit
vip(ga_dt_last_fit_model)

# Modelo final ------------------------------------------------------------

ga_final_dt_model <- ga_dt_wf %>% fit(ga_dataset)


# arquivo de submissao ----------------------------------------------------

ga_submission <- ga_test %>%
  mutate(
    target = predict(ga_dt_last_fit_model, new_data = ., type = "prob")$.pred
  )

write_csv(ga_submission, "ga_submission.csv")

