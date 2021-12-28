#include <cstdio>
int main(){
    int a,i=0,b=0;
    scanf("%d",&a);
    while(i<a){
        int j;
        scanf("%d",&j);
        if(j<2){
            i++
        }
        else if(j%2==0){
            b=b+1;
            i++
        }
        else{
            i++;
        }
    }
    printf("%d",b);
    return 0;
}