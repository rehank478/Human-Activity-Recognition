% [CSI_DATA] = csvimport('jump.csv', 'columns', {'CSI_DATA'});
% CSI_DATA = split(CSI_DATA);
% 
% s = size(CSI_DATA)
% for i =1:s(1)
%     one = (CSI_DATA(i,1));
%     one = split(one, '[');
%     CSI_DATA(i,1) = one(2);
% end
% amp = []
% for i=1:s(1)
%     for j=1:128
%         if mod(j,2)==1
%             real = str2double(CSI_DATA(i,j));
%         else
%             img = str2double(CSI_DATA(i,j));
%             x = sqrt(img*img+real*real);
%             if x==0
%                 amp(i,j/2) = 0.000001;
%             else
%                 amp(i,j/2) = 20*log2(x);
%             end
%         end
%     end
% end
% for i=1:64
%     lev = 5;
%     wname = 'sym6';
%     [dnsig1,c1,l1,threshold_SURE] = wden(amp(:,i),'heursure','s','sln',lev,wname);
%     plot(dnsig1);
%     xlabel('Packet no.');
%     ylabel('CSI Amplitude(dB)');
%     hold on
% end
z = peaks(25);

figure
mesh(z)