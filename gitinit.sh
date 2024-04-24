ssh_address=git@github.com:Ri-chard-Wu/AD-aJLL.git

git config --global user.email "glotigorgeous@gmail.com"
git config --global user.name "Ri-chard-Wu"

git init
git add *
git commit -m "first commit"
git remote add origin $ssh_address
git push -u origin master