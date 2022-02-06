# Hosting files publicly on AWS S3

Amazon S3 is a public object storage where you may host files to be downloaded through your submission scripts. It also provides 5 Gb of Free Tier service that would be more than sufficient for this coursework.

Steps to publicly host any file on S3 bucket:
1. Sign up / Sign in on AWS Console. You can get started here:[https://aws.amazon.com/s3/](https://aws.amazon.com/s3/)
2. From services, select **S3**
3. New users will need to create an S3 bucket that the console will provide you an interface to create. Click the "Create Bucket" button.
4. Now, Choose a unique bucketname (example: slowpoke-umd-nlp) and AWS Region.
5. Uncheck all checkboxes in "Block Public Access settings for this bucket" section.
6. Keep everything default and click "Create Bucket" on the buttom.
7. Now on you dashboard, you should be able to see your newly created bucket. Click on it to view the contents.
8. Currently, it contains nothing. We will be trying to make this entire bucket public so that all items in this bucket can be shared publicly.
9. Select the Permissions tab, and verify that that "Block all public access" is turned OFF.
10. In the bucket policy section click EDIT, and put the following policy json, and click "Save Changes"
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowPublicRead",
            "Effect": "Allow",
            "Principal": {
                "AWS": "*"
            },
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/*"
        }
    ]
}
```
11. Go back to the main S3 dashboard and you should be able to see "Public" in the Access column of your s3 bucket.
12. Now you can open your bucket and upload any file from your local system, or create a directory for organizing your uploads.
13. To get the url that you can use with a `wget` command to download this file on any system, select the file and click "Copy URL"

