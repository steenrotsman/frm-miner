library(tidyverse)
library(xtable)

benchmark <- read_csv("benchmark.csv",
                      col_names = c("Name", "setting", "runtime"),
                      col_types = "fcd")

frm <- benchmark %>%
  filter(substr(setting, 1, 1) == "m") %>%
  separate(setting, c("method", "min_sup", "segment",
                      "alphabet", "min_len", "max_overlap", "fold"),
           sep="_", convert=TRUE) %>%
  filter(min_sup < 0.9) %>%
  group_by(across(Name:max_overlap)) %>%
  summarise(runtime = mean(runtime)) %>%
  group_by(Name) %>%
  summarise(Min=min(runtime), Max=max(runtime))

ostinato <- benchmark %>%
  filter(substr(setting, 1, 1) == "o") %>%
  separate(setting, c("method", "length", "fold"), sep="_", convert=TRUE) %>%
  select(!method) %>%
  group_by(across(Name:length)) %>%
  summarise(runtime = mean(runtime)) %>%
  pivot_wider(names_from = length, values_from = runtime)

res <- left_join(frm, ostinato, by = "Name") %>%
  arrange(`50`)
print(xtable(res), include.rownames = FALSE)