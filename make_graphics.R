require(ggplot2)
require(ggrepel)


changes <- data.frame(read.csv('change_distances.csv',header=T,sep='\t'))

changes.avg <- data.frame(aggregate(cbind(geodist,cophdist,nchange)~grid+feat+var+change+condition,changes,FUN=mean))

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

changes.avg <- data.frame(aggregate(cbind(geodist,cophdist,nchange,pc2)~grid+feat+var+change+condition+niters,changes,FUN=mean))

pdf('distances_plot.pdf')
plot(log(changes.avg$cophdist),log(changes.avg$geodist),type='n',ylab='mean nearest neighbor spatiotemporal distance (log)',xlab='mean cophenetic distance to nearest spatiotemporal neighbor (log)')
text(log(changes.avg$cophdist),log(changes.avg$geodist),changes.avg$condition,cex=changes.avg$pc2-min(changes.avg$pc2)+.5)
dev.off()

changes.avg <- changes.avg[order(changes.avg$pc2),]

changes.avg[,6:10]<-round(changes.avg[,6:10],digits=2)

changes.avg$rank <- c(nrow(changes.avg):1)

colnames(changes.avg) <- c('Grid','Feature','Variant','Change','Condition','niters','Spatiotemporal distance','Cophenetic distance','N changes','PC2','Rank')

changes.avg$Grid <- as.character(changes.avg$Grid)

for (i in 1:nrow(changes.avg)) {
    if (as.numeric(changes.avg[i,]$niters < 80)) {
        gr <- changes.avg[i,]$Grid
        changes.avg[i,]$Grid <- paste(c('*',gr),collapse='')
    }
}

changes.avg$Grid <- as.factor(changes.avg$Grid)

#changes.avg <- changes.avg[,c('Rank','Grid','Feature','Variant','Change','Condition','Spatiotemporal distance','Cophenetic distance','N changes','PC2')]
changes.avg <- changes.avg[,c('Rank','Grid','Feature','Variant','Change','Condition','PC2')]

write.table(changes.avg,'feature_rank.txt',quote=FALSE,sep=' & ',eol='\\\\\n',row.names=FALSE)

require(maps)
require(maptools)

locs <- data.frame(read.csv('phylogeolocs.csv',sep=' ',header=F))
colnames(locs) <- c('lon','lat','type')

pdf('nodelocations.pdf')
#map('world',xlim=range(locs$V1),ylim=range(locs$V2))
#points(locs[locs$V3!='root',]$V1,locs[locs$V3!='root',]$V2,col=locs[locs$V3!='root',]$V3,cex=.5)
#points(locs[locs$V3=='root',]$V1,locs[locs$V3=='root',]$V2,col=locs[locs$V3=='root',]$V3,cex=.75)

ggplot()+borders()+coord_cartesian(xlim=c(-1.5,1.5)+range(locs$lon),ylim=c(-1.5,1.5)+range(locs$lat)) + geom_point(data=locs[locs$type!='root',],aes(x=lon,y=lat,col=type),cex=.25) + geom_point(data=locs[locs$type=='root',],aes(x=lon,y=lat,col=type),cex=1) + theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_blank(), axis.title = element_blank(), axis.ticks = element_blank(), axis.text = element_blank())

dev.off()

IElocs <- data.frame(read.csv('IE_locs.csv',header=F,sep=' '))

colnames(IElocs) <- c('lang','lon','lat')
#IElocs$jitlon <- jitter(IElocs$lon+.5,factor=1000)
#IElocs$jitlat <- jitter(IElocs$lat+.5,factor=1000)


#map('world',xlim=c(-1.5,1.5)+range(IElocs$jitlon),ylim=c(-1.5,1.5)+range(IElocs$jitlat))

#points(IElocs$lon,IElocs$lat,cex=.2)

#labels<-pointLabel(IElocs$lon,IElocs$lat,IElocs$lang,doPlot=F)

#text(labels$x,labels$y,IElocs$lang,cex=.5)

#for (i in 1:nrow(IElocs)) {
##    lines(c(IElocs[i,]$lon,IElocs[i,]$lat),c(IElocs[i,]$jitlon,IElocs[i,]$jitlat))
#    lines(c(IElocs[i,]$lon,IElocs[i,]$jitlon),c(IElocs[i,]$lat,IElocs[i,]$jitlat))
#}


pdf('languagemap.pdf')
ggplot()+borders()+coord_cartesian(xlim=c(-1.5,1.5)+range(IElocs$lon),ylim=c(-1.5,1.5)+range(IElocs$lat))+geom_point(data=IElocs,aes(x=lon,y=lat),cex=2)+geom_text_repel(data=IElocs,aes(x=lon,y=lat,label=lang),cex=2)+theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), axis.line = element_blank(), axis.title = element_blank(), axis.ticks = element_blank(), axis.text = element_blank())
dev.off()
