library(tidyverse)

benchmark <- read_csv("benchmark.csv",
                      col_names = c("name", "setting", "runtime"),
                      col_types = "fcd")

frm <- benchmark %>%
  filter(substr(setting, 1, 1) == "m") %>%
  separate(setting, c("method", "min_sup", "segment",
                      "alphabet", "min_len", "max_overlap", "fold"),
           sep="_", convert=TRUE) %>%
  group_by(across(name:max_overlap)) %>%
  summarise(runtime = mean(runtime)) %>%
  group_by(name) %>%
  summarise(min=min(runtime), max=max(runtime))

ostinato <- benchmark %>%
  filter(substr(setting, 1, 1) == "o") %>%
  separate(setting, c("method", "length", "fold"), sep="_", convert=TRUE) %>%
  select(!method) %>%
  group_by(across(name:length)) %>%
  summarise(runtime = mean(runtime)) %>%
  pivot_wider(names_from = length, values_from = runtime)

res <- inner_join(frm, ostinato, by = "name")