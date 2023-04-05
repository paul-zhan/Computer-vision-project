% ================== importing the data =====================

% create image datastore
imds = imageDatastore('lesionimages');
imgs = readall(imds);

% create image datastore for mask
imds_mask = imageDatastore('masks');
imgs_mask = readall(imds_mask);

% do the gaussian of the image 
for i=1:length(imgs)
   gauss{i} = imgaussfilt(imgs{i}); 
end

% reshape the gauss matrix
gauss = reshape(gauss,200,1);

% ================== image processing =====================

% apply gaussian on the mask 
for i=1:length(imgs_mask)
   gauss_img{i} = imgaussfilt(imgs_mask{i}); 
end


% ================   color ===============================

% create colour histograms
for i=1:length(gauss)
    ch = colourhist(gauss{i});
    allhists(i,:) = ch(:);  % reshape into vector and add to data matrix
end

% do PCA on data
[pcs ,evals ,projdata] = mypca(allhists);
projimgs = projdata(:,1:5);

% ==================== symmetry ==============================

% do the symmetry of the thing   
sysy = sys(imgs_mask);

% =============== border ==========================

%reshape the matrix 
gauss_img = reshape(gauss_img, 200, 1);

for i=1:length(imgs_mask)
   circle{i} = circularity(imgs_mask{i}); 
end

circle = reshape(circle, 200, 1);

for i=1:length(imgs_mask)
   circle{i} = circularity(imgs_mask{i}); 
   circularity_mat(i) = circle{i}(end);

end

circularity_mat = cell2mat(struct2cell(circularity_mat));
circularity_mat = reshape(circularity_mat, 200, 1);

%reshape the matrix 
gauss_img = reshape(gauss_img, 200, 1);

% ================== texture ==========================
for i=1:length(imgs_grey)
    texture{i} = extractLBPFeatures(imgs_grey{i});
    texture_mat(i,:) = texture{1,i}(:);
end


% ========================improving feature ======================

% get a better color feature matrix
color_hist = color_count(allhists);

% reshape the color_hist
color_hist = reshape(color_hist, 200, 1);

% total X2
X2 = cat(2, color_hist, sysy);
X2 = cat(2,X2,pix);
X2 = cat(2,X2, projimgs);
imfeature = cat(2,X2, texture_mat);

%----------------------------   model   -------------------------------


% getting the label for the images
groundtruth = get_label("groundtruth.txt");
 
% set up the svm model 
svm = fitcsvm(imfeature,groundtruth);
 
% performing 10 cross validation
cvsvm = crossval(svm);
 
% obtain the prediction from the datset run
pred = kfoldPredict(cvsvm);

% getting the confusion matrix 
[cm, labels] = confusionmat(groundtruth, pred);
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

function laplacian_filter(dataset)
lf = fspecial('laplacian', 0); % create Laplacian
im = im2double(dataset); % read & convert image
fim = imfilter(im, lf); % filter image
sim = im - fim; % subtract Laplacian image from original
imshow(sim); % show sharpened image
end



