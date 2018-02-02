library(dplyr)
library(jsonlite)
daturl <- "https://kingaa.github.io/pomp/vignettes/twentycities.rda"
datfile <- file.path(tempdir(),"twentycities.rda")
download.file(daturl,destfile=datfile,mode="wb")
load(datfile)

measles %>% 
  mutate(year=as.integer(format(date,"%Y"))) %>%
  subset(town=="London" & year>=1950 & year<1964) %>%
  mutate(time=(julian(date,origin=as.Date("1950-01-01")))/365.25+1950) %>%
  subset(time>1950 & time<1964, select=c(time,cases)) -> dat
demog %>% subset(town=="London",select=-town) -> demogLondon


write_json(toJSON(dat), "/Users/gcgibson/Stein-Variational-Gradient-Descent/dat.json")
write_json(toJSON(demogLondon), "/Users/gcgibson/Stein-Variational-Gradient-Descent/demogLondon.json")


