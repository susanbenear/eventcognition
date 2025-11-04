# Key Steps: 
## 1. Generate bins for each movie according to its length;
## 2. Locate subjects' button presses into the bins;
## 3. Generate the binnized vector for each subject for each movie: e.g., [0,0,0,0,1,0,0,1...]
## 4. Compare all bins to form group norm, and individual correlation with the norm


setwd("/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/")

source("Functions/make.bins_eg.s")
#source("Functions/make.bins.TRs_eg.s")
source("Functions/compare.bins.to.group_eg.s")
source("Functions/bootstrap.agreement.s")


adult_seg <- read.csv("adults_all_segmentation_07.23.22.csv")
adult_bins1000 <- make.bins(adult_seg, 1000)

kid_seg <- read.csv("kids_all_segmentation_07.23.22.csv")
kid_bins1000 <- make.bins(kid_seg, 1000)

#writing these out as csv's to be smoothed
write.csv(adult_bins1000,"/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/alladult_bins_long.csv", row.names = FALSE)
write.csv(kid_bins1000,"/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/allkid_bins_long.csv", row.names = FALSE)

#reshape the df's into wide format using base R
#newadultdf = reshape(nomovieadultdf, idvar = "SubjNum", timevar = "Time", direction = "wide")
#newkiddf = reshape(nomoviekiddf, idvar = "SubjNum", timevar = "Time", direction = "wide")

#SAVE THIS OUT AS A CSV TO BE USED AS AN ARRAY IN PYTHON
#write.csv(newadultdf,"/Users/susanbenear/Google_Drive/Learning_and_Lemurs/Data/Data_Analysis/Event_Segmentation/adultbinsarray.csv", row.names = FALSE)
#write.csv(newkiddf,"/Users/susanbenear/Google_Drive/Learning_and_Lemurs/Data/Data_Analysis/Event_Segmentation/kidbinsarray.csv", row.names = FALSE)


## Note that ScaleCorZ here is inf. With enough data this measure will return sensible values
## as specified in pg. 6 of Zheng et al 2020. This is their segmentation score.
a_compare.bins1000 <-compare.bins.to.group(adult_bins1000)
k_compare.bins1000 <-compare.bins.to.group(kid_bins1000)


yk_compare.bins1000 <-compare.bins.to.group(kid_bins_y1000)
ok_compare.bins1000 <-compare.bins.to.group(kid_bins_o1000)

## kids' scores by adult's norm
k_to.adults_compare.bins1000 <- compare.bins.to.group(bins = kid_bins1000,
                                            bphst = (melt(tapply(adult_bins1000$Bins, 
                                                    list(adult_bins1000$Time,adult_bins1000$Movie),mean), varnames=c("Time","Movie"))))

yk_to.adults_compare.bins1000 <- compare.bins.to.group(bins = kid_bins_y1000,
                                                      bphst = (melt(tapply(adult_bins1000$Bins, 
                                                                           list(adult_bins1000$Time,adult_bins1000$Movie),mean), varnames=c("Time","Movie"))))
ok_to.adults_compare.bins1000 <- compare.bins.to.group(bins = kid_bins_o1000,
                                                      bphst = (melt(tapply(adult_bins1000$Bins, 
                                                                           list(adult_bins1000$Time,adult_bins1000$Movie),mean), varnames=c("Time","Movie"))))

###BOOTSTRAPPING
bins1000 <- cbind(data.frame(Group=c(rep("Adult", dim(adult_bins1000)[1]),
                                        rep("Child", dim(kid_bins1000)[1]))),
                     rbind(adult_bins1000, kid_bins1000))
bag.kappa.KJep2 <- bootagree(bins1000[bins1000$Movie == "KJep2",], R=1000, scorefun=CohensKappa)
bootagree.ci(bag.kappa.KJep2)

## Another useful function is count BPs, which summarizes # of presses for each subject and movie
## This can be used to trimmed down outliers or too few presses. See below for code:

source("Functions/count.bps_eg.s")
k_bps <- count.BPs(kid_seg)
a_bps <- count.BPs(adult_seg)
summary(a_bps)
summary(k_bps)

boxplot(a_bps$NBPs)$out #one pp @ 137
boxplot(k_bps$NBPs)$out #three pp @ 127, 160, & 169

a_bps_trimmed <- a_bps[ which(a_bps$NBPs < 137), ]
k_bps_trimmed <- k_bps[ which(k_bps$NBPs < 127), ]

# export the output of the compare bins function for adults to a csv, so ScaledCorZ 
# can be extracted for other analyses
write.csv(a_compare.bins1000,"/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/adultcomparebinsdf.csv", row.names = FALSE)

# same for kids
write.csv(k_compare.bins1000,"/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/kidcomparebinsdf.csv", row.names = FALSE)

# and kids to the adult norm 
write.csv(k_to.adults_compare.bins1000,"/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/kidtoadultcomparebinsdf.csv", row.names = FALSE)


#save button presses for adults
write.csv(a_bps,"/Users/susanbenear/Google_Drive/Learning_and_Lemurs/Data/Data_Analysis/Event_Segmentation/adultbuttonpresses.csv", row.names = FALSE)

#and for kids
write.csv(k_bps,"/Users/tuh38197/Google_Drive/Learning_and_Lemurs/Data/Data_Analysis/Event_Segmentation/kidbuttonpresses.csv", row.names = FALSE)
