library(ggplot2)
library(dplyr)
library(jsonlite)
library(FRK)
library(mapproj)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

Data_list <- jsonlite::read_json("pacific_end_data.json")
Dens_list <- jsonlite::read_json("pacific_end_dens_est_radial_upper_v20000.json")


Lon <- Lat <- rep(0, length(Data_list))
for(i in 1:length(Data_list)) {
  Lat[i] <- Data_list[[i]][[1]] * 180 / (pi) - 90 
  Lon[i] <- Data_list[[i]][[2]] * 360 / (2*pi) - 180
}
data_df <- data.frame(Lon = Lon, 
                      Lat = Lat)


Lon <- Lat <- dens <- rep(0, length(Dens_list))
for(i in 1:length(Dens_list)) {
  Lat[i] <- Dens_list[[i]][[1]] * 180 / (pi) - 90
  Lon[i] <- Dens_list[[i]][[2]] * 360 / (2*pi) - 180
  dens[i] <- Dens_list[[i]][[3]]
}
dens_df <- data.frame(Lon = Lon, 
                      Lat = Lat,
                      dens = dens)

summary(dens_df[,3])

area <- 6371 ** 2 * 4 / 10000
dens_df[,3] <- dens_df[,3] / area 

subLon <- unique(Lon)[seq(1,length(unique(Lon)))]
subLat <- unique(Lat)[seq(1,length(unique(Lat)))]
dens_df <- filter(dens_df, Lon %in% subLon &
                    Lat %in% subLat)

data(worldmap)


plot_world <- function(lon = 140, show_legend = TRUE) {
  
  g <- ggplot(dens_df) + 
    geom_tile(aes(Lon, Lat, fill = dens)) +
    scale_fill_continuous(low = "#f5f5f5", high = "dark red",
                          name = "Intensity", limit=c(0,5)) +
    geom_path(data = worldmap, aes(x = long, y = lat, group = group)) + 
    coord_map("ortho", orientation =  c(20, lon, 0)) +
    geom_point(data = data_df, aes(x = Lon, y = Lat), 
               colour = "blue", size = 0.001) +
    theme_bw() +
    theme(axis.text.x = element_blank()) +
    theme(axis.text.y = element_blank()) +
    theme(axis.ticks = element_blank()) +
    ylab("") + xlab("")
  
  
  if(!show_legend) g <- g + theme(legend.position = "none",
                                  legend.text=element_text(size=.1))  
  g
}

g1 <- plot_world(lon = 240, show_legend = TRUE)
ggsave(g1, file = "pacific_end_dens1_est_radial_v20000_upper.png", 
       width = 5.6, height = 4, dpi = 300)
