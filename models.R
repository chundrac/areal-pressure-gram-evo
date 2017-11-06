changes <- data.frame(read.csv('change_distances.csv',header=T,sep='\t'))

lenlevels <- function(x) {
    return(length(levels(as.factor(x))))
}
num.iters <- data.frame(aggregate(iter~grid+feat+var+change,changes,FUN=lenlevels))


changes$niters <- rep(0,nrow(changes))
for (i in 1:nrow(changes)) {
    changes[i,]$niters <- num.iters[num.iters$grid==changes[i,]$grid & num.iters$feat==changes[i,]$feat & num.iters$var==changes[i,]$var & num.iters$change==changes[i,]$change,]$iter }


changes[,c('geodist','cophdist')] <- log(changes[,c('geodist','cophdist')])
changes$pc2 <- rep(0,nrow(changes))
for (i in 0:17) {
    pc <- prcomp(changes[changes$tree==i,c('cophdist','geodist')])$x[,2]
    if (cor.test(pc,changes[changes$tree==i,]$geodist)[4] > 0) {
        pc <- -pc
    }
#    print (cor.test(changes[changes$tree==i,]$cophdist,changes[changes$tree==i,]$geodist))
#    print (summary(prcomp(changes[changes$tree==i,c('cophdist','geodist')])))
    changes[changes$tree==i,]$pc2 <- pc
}



changes$SNvsC <- rep(0,nrow(changes))
changes$SvsN <- rep(0,nrow(changes))

changes[changes$condition=='C',]$SNvsC <- -2/3
changes[changes$condition=='C',]$SvsN <- 0
changes[changes$condition=='N',]$SNvsC <- 1/3
changes[changes$condition=='N',]$SvsN <- -1/2
changes[changes$condition=='S',]$SNvsC <- 1/3
changes[changes$condition=='S',]$SvsN <- 1/2

changes$featvar <- paste(changes$feat,changes$var,sep='+')

require(lme4)
require(MASS)

#model.lmer <- lmer(pc2 ~ SNvsC + SvsN + (1|iter) + (1|tree) + (1|grid) + (1|featvar) + (1|change)
#                   + (SvsN|tree) + (NvsC|tree)
#                   , changes)
#model.lmer
#dropterm(model.lmer,test='Chisq')


changes.small <- changes[changes$niters > 80,]
model.lmer <- lmer(pc2 ~ SNvsC + SvsN + (1|iter) + (1|tree) + (1|grid) + (1|featvar) + (1|change)
# + (SvsN|tree) + (NvsC|tree)
, changes.small)
model.lmer
model.sig <- dropterm(model.lmer,test='Chisq')
model.sig
pvals <- data.frame(model.sig)[2:3,4]
cat(pvals,file='modelpvals.txt',sep='\t',fill=T,append=T)
