function outVal = scalinglaw_Tonini_WC(typeScal,M)
%
% From email Paolo 24/04/2019
%
% def fault_length(m):
%     # fault length
%     a=-2.440
%     b=0.590
%     length = np.power(10, (a + b*m))
%     return float(length)
% def fault_width(m):
%     # fault width
%     a=-1.010
%     b=0.320
%     width = np.power(10, (a + b*m))
%     return float(width)
%     
    
    if strcmp(typeScal,'M2L')
      a =-2.440;
      b =0.590;
      outVal = 10.^(a+b*M);
    elseif strcmp(typeScal,'M2W') 
      a=-1.010;
      b=0.320;
      outVal = 10.^(a+b*M);
    else
      disp(['Type ' typeScal ' not recongnized!!']);
    end  

end

    
    

