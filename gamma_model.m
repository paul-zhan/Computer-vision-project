% ================== importing the data =====================

% create image datastore
imds = imageDatastore('lesionimages');
imgs = readall(imds);

% create image datastore for mask
imds_mask = imageDatastore('masks');
imgs_mask = readall(imds_mask);


% ================== image processing =====================

% create image grey scale 
for i=1:length(imgs)
    imgs_grey{i} = rgb2gray(imgs{i});
end

imgs_grey = reshape(imgs_grey, 200, 1);

% create gamma image 
for i=1:length(imgs)
    imgs_gamma{i} = gamma(imgs{i}, 0.25);
end

imgs_gamma = reshape(imgs_gamma, 200, 1);

% ================   color ===============================

% create colour histograms
for i=1:length(imgs_gamma)
    ch = colourhist(imgs_gamma{i});
    allhists(i,:) = ch(:);  % reshape into vector and add to data matrix
end

% do PCA on data
[pcs ,evals ,projdata] = mypca(allhists);
projimgs = projdata(:,1:5);

% ==================== symmetry ==============================

% do the symmetry of the thing   
sysy = sys(imgs_mask);
%sysy_average = averages(sysy);




% =============== border ==========================
% apply gaussian on the mask 
for i=1:length(imgs_mask)
   gauss_img{i} = imgaussfilt(imgs_mask{i}); 
end

%reshape the matrix 
gauss_img = reshape(gauss_img, 200, 1);

% getting the difference between the gaussian image and the mask image 
for  i=1:length(imgs_mask)
    image_diff{i} = imabsdiff(gauss_img{i}, imgs_mask{i});
end


% reshape the matrix image_diff
image_diff = reshape(image_diff, 200, 1);

% =====================color invariant =========================
% color invariant
%image_GW = im2uint8(image_GW);


% ================== texture ==========================
for i=1:length(imgs_grey)
    texture{i} = extractLBPFeatures(imgs_grey{i});
    texture_mat(i,:) = texture{1,i}(:);
end

% ========================improving feature ======================


% counting the pixel
pix = counting(image_diff);



% new X matrix combining color and symmetry
%X = cat(2, allhists ,sysy_average);
% concatenate features
%X = cat(2,X, pix);

% get a better color feature matrix
color_hist = color_count(allhists);


% reshape the color_hist
color_hist = reshape(color_hist, 200, 1);


% total X2
X2 = cat(2, color_hist, sysy);
X2 = cat(2,X2,pix);
X2 = cat(2,X2, texture_mat);
X2 = movmean(X2, 15);



%----------------------------   model   -------------------------------


% getting the label for the images
label = get_label("groundtruth.txt");
 
% set up the svm model 
svm = fitcsvm(projimgs,label);
 
% performing 10 cross validation
cvsvm = crossval(svm);
 
% obtain the prediction from the datset run
pred = kfoldPredict(cvsvm);

% getting the confusion matrix 
[cm, labels] = confusionmat(label, pred, 'order',{'malignant','benign'});
[accuracy , missclassification_rate, precision, recall] = result(cm);




%-----------------------   function   ----------------------------


 
function label = get_label(text_file)
     string = readlines(text_file);
     string2 = string(1:end-1);
     label = [];
     
     for i = 1:length(string2)
         label = extractAfter(string2, ',');
     end
end
 
function [pcs, evals, projdata] = mypca(data)
    c = cov(data); % covariance matrix
    [v ,d] = eig(c); % get eigenvectors
    d = diag(d);
    [~, ind] = sort(d, 'descend'); % sort eigenvalues
    pcs = v(:,ind);
    evals = d(ind);
    projdata = data * pcs; % project onto PCA space
end

function H = colourhist(image)
% generate 8x8x8 RGB colour histogram from image
noBins = 8; % 8 bins (along each dimension)
binWidth = 256 / noBins; % width of each bin
H = zeros(noBins, noBins, noBins); % empty histogram to start with
[n ,m ,d] = size(image);
data = reshape(image, n*m, d); % reshape image into 2-d matrix with one row per pixel
ind = floor(double(data) / binWidth) + 1; % calculate into which bin each pixel falls
for i=1:length(ind)
H(ind(i,1), ind(i,2), ind(i,3)) = H(ind(i,1), ind(i,2), ind(i,3))+ 1; % increment bin
end
H = H / sum(sum(sum(H))); % normalise histogram
end

function [accuracy , missclassification_rate, precision, recall] = result(matrix)
    accuracy  = (matrix(1,1) + matrix(2,2))/(matrix(1,1)+matrix(1,2)+matrix(2,1)+matrix(2,2));
    missclassification_rate = (matrix(1,2) + matrix(2,1))/(matrix(1,1)+matrix(1,2)+matrix(2,1)+matrix(2,2));
    precision = matrix(1,1)/(matrix(1,1) + matrix(1,2));
    recall = matrix(1,1)/(matrix(1,1) + matrix(2,1));
    
end


function sys = sys(list)
    sys = zeros(200,6);
    for i = 1:200
        img = list{i};
        for j = 1:6
            img = imrotate(img, 60); 
            sys(i,j) = sum((img & fliplr(img)))/ sum((img| fliplr(img)));      
        end 
        
    end
end

function counting = counting(list)
counting = zeros(200,1);
    for i= 1:length(list)
        count = 0;
        [w,l] = size(list{i});
        for j =1:w
            for k=1:l
                if list{i}(j,k)> 0
                    count = count +1;
                end
            end
        end
        counting(i) = count;    
    end
end 



function colors = color_count(matrix)
    [l, w ] = size(matrix);
    for i =1:l
        count = 0;
        for j=1:w
            if matrix(i,j)> 0 
                count = count +1;
            end  
        end 
        colors(i) = count;
    end
end     

function nim = gamma(im, g)
    % perform gamma correction/enhancement
    nim = im2uint8(im2double(im).^g);
end

function count_matrix = averages(matrix) 
   count_matrix = zeros(200,1);
   
   [w,l] = size(matrix);
   for i= 1:w
      count = 0;
      for j=1:l
          if matrix(i,j)> 0.7
              count = count + 1;
          end
      end
      count_matrix(i) = count; 
   end           
end 

function [image_GW] = rgb2gw(image_RGB)
    % Greyworld colour constancy & white balancing
    image_RGB = im2double(image_RGB);
    illum = squeeze(mean(image_RGB, [1 2])); % estimate illuminant as mean of channels
    illum = illum / norm(illum, 2); % normalise to unit length
    for i=1:size(image_RGB, 3)
        image_GW(:,:,i) = min(1, (1/sqrt(3))/illum(i) * image_RGB(:,:,i));
    end
end



