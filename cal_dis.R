PL <- Community_data2 %>% mutate(
  Severity = (Disease_0*0 + Disease_1*1 +Disease_2*2+Disease_3*3+ Disease_4*4+
                Disease_5*5)/
    6/(Disease_0+Disease_1+Disease_2+Disease_3+Disease_4+Disease_5)) %>% 
  groupby(Treatment,Site,plot) %>% summarise(PL= sum(Severity*species_biomass,na.rm=True)/mean(Plot_Biomass))
