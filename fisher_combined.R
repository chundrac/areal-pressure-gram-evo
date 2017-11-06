pvals <- data.frame(read.csv('modelpvals.txt',sep='\t',header=F))
colnames(pvals)<-c('SNvsC','SvsN')

print('Results of Fisher combined probability test')

chi.SNvsC <- -2*sum(log(pvals$SNvsC))
print(paste('SNvsC',pchisq(chi.SNvsC,length(pvals$SNvsC)*2,lower.tail=F),sep=': '))

chi.SvsN <- -2*sum(log(pvals$SvsN))
print(paste('SvsN',pchisq(chi.SvsN,length(pvals$SvsN)*2,lower.tail=F),sep=': '))
